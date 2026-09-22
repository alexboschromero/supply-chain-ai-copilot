
import io, os, math, html as html_lib, zipfile
from datetime import date, timedelta, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

try:
    from openai import OpenAI, AuthenticationError, APIError, RateLimitError
except Exception:
    OpenAI = None
    AuthenticationError = Exception
    APIError = Exception
    RateLimitError = Exception

try:
    import xlsxwriter
except Exception:
    xlsxwriter = None

st.set_page_config(
    page_title="Supply Chain AI Copilot",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Core analytics
# -----------------------------
REQUIRED = [
    "SKU","Description","Supplier","Year","Month","Sales","Stock",
    "Open_PO","Lead_Time_Days","MOQ","Unit_Cost"
]

@st.cache_data
def sample_data():
    return pd.read_csv(Path(__file__).parent/"data"/"sample_history.csv")


def read_uploaded(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))

def normalize_columns(df):
    aliases = {
        "sku": "SKU",
        "item": "SKU",
        "item_code": "SKU",
        "article": "SKU",
        "description": "Description",
        "product": "Description",
        "supplier": "Supplier",
        "vendor": "Supplier",
        "year": "Year",
        "month": "Month",
        "sales": "Sales",
        "units_sold": "Sales",
        "demand": "Sales",
        "stock": "Stock",
        "inventory": "Stock",
        "open_po": "Open_PO",
        "open_po_units": "Open_PO",
        "lead_time": "Lead_Time_Days",
        "lead_time_days": "Lead_Time_Days",
        "lt_days": "Lead_Time_Days",
        "moq": "MOQ",
        "unit_cost": "Unit_Cost",
        "cost": "Unit_Cost",
    }
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[c] = aliases[key]
    return df.rename(columns=rename)


def _periods_from_raw(raw):
    if not all(c in raw.columns for c in ["Year","Month"]):
        return []
    tmp = raw[["Year","Month"]].copy()
    tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce")
    tmp["Month"] = pd.to_numeric(tmp["Month"], errors="coerce")
    tmp = tmp.dropna().drop_duplicates().sort_values(["Year","Month"])
    return [f"{int(r.Year)}-{int(r.Month):02d}" for _, r in tmp.iterrows()]

def _period_tuple(period_text):
    y, m = str(period_text).split("-")
    return int(y), int(m)

def _period_cutoff_mask(raw, period_text):
    y, m = _period_tuple(period_text)
    years = pd.to_numeric(raw["Year"], errors="coerce")
    months = pd.to_numeric(raw["Month"], errors="coerce")
    return (years < y) | ((years == y) & (months <= m))

def _period_exact_mask(raw, period_text):
    y, m = _period_tuple(period_text)
    years = pd.to_numeric(raw["Year"], errors="coerce")
    months = pd.to_numeric(raw["Month"], errors="coerce")
    return (years == y) & (months == m)

def _snapshot_for_period(raw, period_text):
    x = raw[_period_exact_mask(raw, period_text)].copy()
    if x.empty:
        return pd.DataFrame(columns=["SKU","Description","Supplier","Sales_Actual","Stock_Actual","Open_PO_Actual"])
    x = x.sort_values(["SKU"])
    return x.groupby("SKU", as_index=False).agg(
        Description=("Description","first"),
        Supplier=("Supplier","first"),
        Sales_Actual=("Sales","sum"),
        Stock_Actual=("Stock","last"),
        Open_PO_Actual=("Open_PO","last"),
    )

def _safe_pct_delta(current, previous):
    current = np.asarray(pd.to_numeric(current, errors="coerce"), dtype=float)
    previous = np.asarray(pd.to_numeric(previous, errors="coerce"), dtype=float)
    return np.where(
        np.abs(previous) > 1e-9,
        (current - previous) / np.abs(previous) * 100,
        np.where(np.abs(current) > 1e-9, 100.0, 0.0)
    )

def _change_classification(row):
    worsen = 0
    improve = 0
    if row["Service_Risk_Delta"] > 0: worsen += 2
    elif row["Service_Risk_Delta"] < 0: improve += 2
    if row["Purchase_Value_Delta"] > 0: worsen += 1
    elif row["Purchase_Value_Delta"] < 0: improve += 1
    if row["Days_Cover_Delta"] < -1: worsen += 1
    elif row["Days_Cover_Delta"] > 1: improve += 1
    if row["Current_Action"] == "BUY_NOW" and row["Previous_Action"] != "BUY_NOW": worsen += 2
    elif row["Previous_Action"] == "BUY_NOW" and row["Current_Action"] != "BUY_NOW": improve += 2
    if worsen >= 3: return "WORSENED"
    if improve >= 3: return "IMPROVED"
    if worsen == 2 or improve == 2: return "WATCH"
    return "STABLE"

def _change_reason(row):
    reasons = []
    if row["Service_Risk_Delta"] > 0: reasons.append("service risk increased")
    elif row["Service_Risk_Delta"] < 0: reasons.append("service risk decreased")
    if row["Days_Cover_Delta"] < -1: reasons.append("coverage fell")
    elif row["Days_Cover_Delta"] > 1: reasons.append("coverage improved")
    if row["Purchase_Value_Delta"] > 0: reasons.append("purchase exposure increased")
    elif row["Purchase_Value_Delta"] < 0: reasons.append("purchase exposure decreased")
    if row["Action_Changed"]:
        reasons.append(f"action changed {row['Previous_Action']} → {row['Current_Action']}")
    return "; ".join(reasons) if reasons else "No material decision change."

def build_period_comparison(raw, safety_days, service, current_period=None, previous_period=None):
    periods = _periods_from_raw(raw)
    meta = {
        "available_periods": periods, "current_period": current_period, "previous_period": previous_period,
        "has_comparison": False, "purchase_delta": 0.0, "purchase_delta_pct": 0.0,
        "service_risk_delta": 0.0, "service_risk_delta_pct": 0.0,
        "excess_delta": 0.0, "excess_delta_pct": 0.0, "critical_delta": 0,
        "action_changes": 0, "worsened": 0, "improved": 0, "watch": 0,
        "purchase_current": 0.0, "service_risk_current": 0.0, "excess_current": 0.0,
    }
    if len(periods) < 2:
        return pd.DataFrame(), None, None, meta
    if current_period not in periods:
        current_period = periods[-1]
    current_idx = periods.index(current_period)
    valid_previous = periods[:current_idx]
    if not valid_previous:
        return pd.DataFrame(), None, None, meta
    if previous_period not in valid_previous:
        previous_period = valid_previous[-1]

    raw_current = raw[_period_cutoff_mask(raw, current_period)].copy()
    raw_previous = raw[_period_cutoff_mask(raw, previous_period)].copy()
    current_a = analyze(raw_current, safety_days, service)
    previous_a = analyze(raw_previous, safety_days, service)

    cols = ["SKU","Status","Action","Days_Cover","Recommended_Order","Purchase_Value",
            "Service_Risk_Value","Excess_Inventory_Value","Decision_Score","Forecast_Next_Month"]
    cur = current_a[cols].rename(columns={
        "Status":"Current_Status","Action":"Current_Action","Days_Cover":"Current_Days_Cover",
        "Recommended_Order":"Current_Recommended_Order","Purchase_Value":"Current_Purchase_Value",
        "Service_Risk_Value":"Current_Service_Risk_Value","Excess_Inventory_Value":"Current_Excess_Value",
        "Decision_Score":"Current_Decision_Score","Forecast_Next_Month":"Current_Forecast"})
    prev = previous_a[cols].rename(columns={
        "Status":"Previous_Status","Action":"Previous_Action","Days_Cover":"Previous_Days_Cover",
        "Recommended_Order":"Previous_Recommended_Order","Purchase_Value":"Previous_Purchase_Value",
        "Service_Risk_Value":"Previous_Service_Risk_Value","Excess_Inventory_Value":"Previous_Excess_Value",
        "Decision_Score":"Previous_Decision_Score","Forecast_Next_Month":"Previous_Forecast"})
    comp = cur.merge(prev, on="SKU", how="outer")
    for c in ["Current_Status","Current_Action","Previous_Status","Previous_Action"]:
        comp[c] = comp[c].fillna("NONE")
    for c in comp.columns:
        if c != "SKU" and c not in {"Current_Status","Current_Action","Previous_Status","Previous_Action"}:
            comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0)

    snap = _snapshot_for_period(raw, current_period).merge(
        _snapshot_for_period(raw, previous_period),
        on="SKU", how="outer", suffixes=("_Current","_Previous")
    )
    for c in ["Description_Current","Description_Previous","Supplier_Current","Supplier_Previous"]:
        if c in snap.columns:
            snap[c] = snap[c].fillna("")
    comp = comp.merge(snap, on="SKU", how="outer")
    comp["Description"] = np.where(comp["Description_Current"].astype(str).str.len() > 0, comp["Description_Current"], comp["Description_Previous"])
    comp["Supplier"] = np.where(comp["Supplier_Current"].astype(str).str.len() > 0, comp["Supplier_Current"], comp["Supplier_Previous"])
    for c in ["Sales_Actual_Current","Sales_Actual_Previous","Stock_Actual_Current","Stock_Actual_Previous","Open_PO_Actual_Current","Open_PO_Actual_Previous"]:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0)

    comp["Sales_Delta_Pct"] = _safe_pct_delta(comp["Sales_Actual_Current"], comp["Sales_Actual_Previous"])
    comp["Stock_Delta_Pct"] = _safe_pct_delta(comp["Stock_Actual_Current"], comp["Stock_Actual_Previous"])
    comp["Open_PO_Delta"] = comp["Open_PO_Actual_Current"] - comp["Open_PO_Actual_Previous"]
    comp["Days_Cover_Delta"] = comp["Current_Days_Cover"] - comp["Previous_Days_Cover"]
    comp["Recommended_Order_Delta"] = comp["Current_Recommended_Order"] - comp["Previous_Recommended_Order"]
    comp["Purchase_Value_Delta"] = comp["Current_Purchase_Value"] - comp["Previous_Purchase_Value"]
    comp["Service_Risk_Delta"] = comp["Current_Service_Risk_Value"] - comp["Previous_Service_Risk_Value"]
    comp["Excess_Value_Delta"] = comp["Current_Excess_Value"] - comp["Previous_Excess_Value"]
    comp["Decision_Score_Delta"] = comp["Current_Decision_Score"] - comp["Previous_Decision_Score"]
    comp["Action_Changed"] = comp["Current_Action"] != comp["Previous_Action"]
    comp["Action_Transition"] = comp["Previous_Action"].astype(str) + " → " + comp["Current_Action"].astype(str)
    comp["Change_Classification"] = comp.apply(_change_classification, axis=1)
    comp["Change_Reason"] = comp.apply(_change_reason, axis=1)
    comp["Change_Score"] = (
        (comp["Service_Risk_Delta"] > 0).astype(int) * 2
        + (comp["Purchase_Value_Delta"] > 0).astype(int)
        + (comp["Days_Cover_Delta"] < -1).astype(int)
        + ((comp["Current_Action"]=="BUY_NOW") & (comp["Previous_Action"]!="BUY_NOW")).astype(int) * 2
        - (comp["Service_Risk_Delta"] < 0).astype(int) * 2
        - (comp["Purchase_Value_Delta"] < 0).astype(int)
        - (comp["Days_Cover_Delta"] > 1).astype(int)
        - ((comp["Previous_Action"]=="BUY_NOW") & (comp["Current_Action"]!="BUY_NOW")).astype(int) * 2
    )
    comp = comp.sort_values(["Change_Score","Service_Risk_Delta","Purchase_Value_Delta"], ascending=[False,False,False]).reset_index(drop=True)
    comp["Priority"] = np.arange(1, len(comp)+1)

    purchase_current = float(current_a["Purchase_Value"].sum())
    purchase_previous = float(previous_a["Purchase_Value"].sum())
    service_current = float(current_a["Service_Risk_Value"].sum())
    service_previous = float(previous_a["Service_Risk_Value"].sum())
    excess_current = float(current_a["Excess_Inventory_Value"].sum())
    excess_previous = float(previous_a["Excess_Inventory_Value"].sum())
    meta.update({
        "current_period": current_period, "previous_period": previous_period, "has_comparison": True,
        "purchase_current": purchase_current, "purchase_previous": purchase_previous,
        "purchase_delta": purchase_current - purchase_previous,
        "purchase_delta_pct": float(_safe_pct_delta(purchase_current, purchase_previous)),
        "service_risk_current": service_current, "service_risk_previous": service_previous,
        "service_risk_delta": service_current - service_previous,
        "service_risk_delta_pct": float(_safe_pct_delta(service_current, service_previous)),
        "excess_current": excess_current, "excess_previous": excess_previous,
        "excess_delta": excess_current - excess_previous,
        "excess_delta_pct": float(_safe_pct_delta(excess_current, excess_previous)),
        "critical_current": int((current_a["Status"]=="🔴 CRITICAL").sum()),
        "critical_previous": int((previous_a["Status"]=="🔴 CRITICAL").sum()),
        "critical_delta": int((current_a["Status"]=="🔴 CRITICAL").sum() - (previous_a["Status"]=="🔴 CRITICAL").sum()),
        "action_changes": int(comp["Action_Changed"].sum()),
        "worsened": int((comp["Change_Classification"]=="WORSENED").sum()),
        "improved": int((comp["Change_Classification"]=="IMPROVED").sum()),
        "watch": int((comp["Change_Classification"]=="WATCH").sum()),
    })
    return comp, current_a, previous_a, meta

def agent_change_monitor_tool(comparison, meta=None):
    if comparison is None or comparison.empty:
        return {"name":"change_monitor","purpose":"Compare two planning periods.","kpis":{"available":False},"rows":[]}
    meta = meta or {}
    x = comparison.head(10)
    return {
        "name":"change_monitor",
        "purpose":"Explain what materially changed between the selected periods and which SKUs need attention.",
        "kpis":{
            "available":True,
            "current_period":meta.get("current_period"),
            "previous_period":meta.get("previous_period"),
            "purchase_delta":meta.get("purchase_delta",0),
            "service_risk_delta":meta.get("service_risk_delta",0),
            "excess_delta":meta.get("excess_delta",0),
            "critical_delta":meta.get("critical_delta",0),
            "action_changes":meta.get("action_changes",0),
            "worsened":meta.get("worsened",0),
            "improved":meta.get("improved",0),
        },
        "rows":x[[
            "SKU","Description","Supplier","Previous_Action","Current_Action","Action_Transition",
            "Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
            "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta",
            "Sales_Delta_Pct","Change_Classification","Change_Reason"
        ]].round(2).to_dict("records")
    }

def data_quality_report(df):
    rows = []
    for col in REQUIRED:
        if col not in df.columns:
            rows.append({
                "Category": "Structure", "Check": f"Required field: {col}",
                "Status": "CRITICAL", "Count": 1,
                "Details": "Required field not found."
            })
            continue
        nulls = int(df[col].isna().sum())
        rows.append({
            "Category": "Completeness", "Check": col,
            "Status": "OK" if nulls == 0 else "WARNING",
            "Count": nulls,
            "Details": "No missing values." if nulls == 0 else f"{nulls} missing values."
        })

    for col in ["Sales","Stock","Open_PO","Lead_Time_Days","MOQ","Unit_Cost"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            bad = int((values < 0).sum())
            rows.append({
                "Category": "Validity",
                "Check": f"Negative values: {col}",
                "Status": "OK" if bad == 0 else "WARNING",
                "Count": bad,
                "Details": "No negative values." if bad == 0 else f"{bad} rows contain negative values."
            })

    if "Month" in df.columns:
        month_num = pd.to_numeric(df["Month"], errors="coerce")
        bad_month = int(((month_num < 1) | (month_num > 12)).sum())
        rows.append({
            "Category": "Validity", "Check": "Month range",
            "Status": "OK" if bad_month == 0 else "CRITICAL",
            "Count": bad_month,
            "Details": "Months are 1–12." if bad_month == 0 else f"{bad_month} rows have invalid month values."
        })

    if all(c in df.columns for c in ["SKU","Year","Month"]):
        dupes = int(df.duplicated(subset=["SKU","Year","Month"], keep=False).sum())
        rows.append({
            "Category": "Uniqueness", "Check": "SKU-Year-Month",
            "Status": "OK" if dupes == 0 else "WARNING",
            "Count": dupes,
            "Details": "No duplicated SKU-period rows." if dupes == 0 else f"{dupes} rows participate in duplicated SKU-period combinations."
        })

    if "SKU" in df.columns:
        empty_sku = int(df["SKU"].astype(str).str.strip().eq("").sum())
        rows.append({
            "Category": "Completeness", "Check": "Blank SKU",
            "Status": "OK" if empty_sku == 0 else "CRITICAL",
            "Count": empty_sku,
            "Details": "No blank SKUs." if empty_sku == 0 else f"{empty_sku} blank SKU values."
        })

    if "Supplier" in df.columns:
        empty_supplier = int(df["Supplier"].astype(str).str.strip().eq("").sum())
        rows.append({
            "Category": "Completeness", "Check": "Blank Supplier",
            "Status": "OK" if empty_supplier == 0 else "WARNING",
            "Count": empty_supplier,
            "Details": "No blank suppliers." if empty_supplier == 0 else f"{empty_supplier} blank supplier values."
        })

    return pd.DataFrame(rows)

def data_quality_summary(df, dq):
    critical = int((dq["Status"] == "CRITICAL").sum())
    warnings = int((dq["Status"] == "WARNING").sum())
    ok = int((dq["Status"] == "OK").sum())
    latest_period = "—"
    if all(c in df.columns for c in ["Year","Month"]):
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        mask = y.notna() & m.notna()
        if mask.any():
            latest_period = f"{int(y[mask].max())}-{int(m[mask].max()):02d}"
    return {
        "rows": int(len(df)),
        "skus": int(df["SKU"].nunique()) if "SKU" in df.columns else 0,
        "suppliers": int(df["Supplier"].nunique()) if "Supplier" in df.columns else 0,
        "critical": critical, "warnings": warnings, "ok": ok,
        "latest_period": latest_period, "checks": int(len(dq))
    }



def build_planning_agent(a, comparison=None, comparison_meta=None):
    x = a.copy()

    urgency = x["Action"].map({
        "BUY_NOW": 100, "CONFIRM_PO": 85, "REVIEW": 60,
        "DO_NOT_BUY": 45, "MONITOR": 10
    }).fillna(0)

    def norm(s):
        m = float(s.max()) if len(s) else 0.0
        return s / m if m > 0 else s * 0

    risk_value = pd.to_numeric(x["Service_Risk_Value"], errors="coerce").fillna(0)
    purchase_value = pd.to_numeric(x["Purchase_Value"], errors="coerce").fillna(0)
    decision_score = pd.to_numeric(x["Decision_Score"], errors="coerce").fillna(0)
    cover_gap = pd.to_numeric(x["Lead_Time_Gap_Days"], errors="coerce").fillna(0)

    x["Execution_Score"] = (
        urgency * 0.40
        + norm(risk_value) * 35
        + norm(purchase_value) * 15
        + norm(decision_score) * 10
        + np.maximum(-cover_gap, 0) * 2
    )

    def execution_meta(r):
        if r["Action"] == "CONFIRM_PO":
            return 1, "Confirm supplier ETA / quantity"
        if r["Action"] == "BUY_NOW":
            return 2, "Release / validate purchase"
        if r["Action"] == "DO_NOT_BUY":
            return 3, "Block replenishment"
        if r["Action"] == "REVIEW":
            return 4, "Review policy / planning parameters"
        return 5, "Monitor"

    steps = x.apply(execution_meta, axis=1, result_type="expand")
    x["Execution_Step"] = steps[0].astype(int)
    x["Execution_Task"] = steps[1]

    x["Dependency"] = np.where(
        x["Action"].eq("BUY_NOW") & x["Open_PO"].gt(0),
        "Validate existing PO first",
        np.where(x["Action"].eq("CONFIRM_PO"),
                 "Supplier confirmation required",
                 "No dependency")
    )

    x["Expected_Service_Risk_Addressed"] = np.where(
        x["Action"].isin(["BUY_NOW", "CONFIRM_PO"]),
        np.minimum(
            risk_value,
            np.maximum(
                pd.to_numeric(x["Recommended_Order"], errors="coerce").fillna(0)
                * pd.to_numeric(x["Unit_Cost"], errors="coerce").fillna(0),
                0
            )
        ),
        0
    )

    x["Blocked_Purchase_Exposure"] = np.where(
        x["Action"].eq("DO_NOT_BUY"), purchase_value, 0
    )

    x["Planning_Rationale"] = x.apply(
        lambda r: (
            f"{r['Action']}: {r['Description']}. "
            f"Coverage {r['Days_Cover']:.1f}d vs lead time {r['Lead_Time_Days']:.1f}d. "
            f"Service-risk exposure €{r['Service_Risk_Value']:,.0f}. "
            f"Purchase exposure €{r['Purchase_Value']:,.0f}. "
            f"{r['Dependency']}."
        ),
        axis=1
    )

    x = x.sort_values(
        ["Execution_Step","Execution_Score","Service_Risk_Value","Decision_Score"],
        ascending=[True,False,False,False]
    ).reset_index(drop=True)
    x["Execution_Priority"] = np.arange(1, len(x) + 1)

    meta = {
        "action_count": int(len(x)),
        "immediate_count": int(x["Action"].isin(["BUY_NOW","CONFIRM_PO"]).sum()),
        "purchase_value": float(x["Purchase_Value"].sum()),
        "service_risk_value": float(x["Service_Risk_Value"].sum()),
        "service_risk_addressed": float(x["Expected_Service_Risk_Addressed"].sum()),
        "blocked_purchase_exposure": float(x["Blocked_Purchase_Exposure"].sum()),
        "buy_now_count": int((x["Action"]=="BUY_NOW").sum()),
        "confirm_po_count": int((x["Action"]=="CONFIRM_PO").sum()),
        "block_count": int((x["Action"]=="DO_NOT_BUY").sum()),
        "review_count": int((x["Action"]=="REVIEW").sum()),
    }
    if comparison_meta and comparison_meta.get("has_comparison"):
        meta["change_worsened"] = int(comparison_meta.get("worsened", 0))
        meta["change_improved"] = int(comparison_meta.get("improved", 0))
        meta["action_changes"] = int(comparison_meta.get("action_changes", 0))
    else:
        meta["change_worsened"] = 0
        meta["change_improved"] = 0
        meta["action_changes"] = 0

    return x, meta

def agent_planning_tool(a, comparison=None, comparison_meta=None):
    planning, meta = build_planning_agent(a, comparison, comparison_meta)
    return {
        "name": "planning_agent",
        "purpose": "Create an ordered execution sequence from the decision engine.",
        "kpis": meta,
        "rows": planning.head(12)[[
            "Execution_Priority","SKU","Description","Supplier","Action",
            "Execution_Task","Dependency","Action_Timing","Days_Cover",
            "Lead_Time_Days","Recommended_Order","Purchase_Value",
            "Service_Risk_Value","Expected_Service_Risk_Addressed",
            "Blocked_Purchase_Exposure","Decision_Confidence","Planning_Rationale"
        ]].round(2).to_dict("records")
    }

def build_action_plan(a):
    x = a.sort_values("Decision_Score", ascending=False).copy()
    plan = []
    for _, r in x.iterrows():
        if r["Action"] == "BUY_NOW":
            action = f"BUY {r['Recommended_Order']:.0f} units"
            owner = "Planner"
            deadline = "Today"
            reason = f"Coverage {r['Days_Cover']:.1f}d < lead time {r['Lead_Time_Days']:.0f}d"
        elif r["Action"] == "CONFIRM_PO":
            action = f"CONFIRM OPEN PO ({r['Open_PO']:.0f} units)"
            owner = "Planner / Buyer"
            deadline = "This week"
            reason = f"Existing PO should cover requirement; validate ETA"
        elif r["Action"] == "REVIEW":
            action = "REVIEW REPLENISHMENT POLICY"
            owner = "Planner"
            deadline = "Next cycle"
            reason = "Below target but not yet a critical stockout"
        elif r["Action"] == "DO_NOT_BUY":
            action = "BLOCK NEW REPLENISHMENT"
            owner = "Planner"
            deadline = "Immediate"
            reason = f"Coverage {r['Days_Cover']:.1f}d indicates excess"
        else:
            action = "MONITOR"
            owner = "Planner"
            deadline = "Routine"
            reason = "Within current policy"
        plan.append({
            "Priority": len(plan) + 1,
            "SKU": r["SKU"],
            "Description": r["Description"],
            "Supplier": r["Supplier"],
            "Action": action,
            "Owner": owner,
            "Deadline": deadline,
            "Reason": reason,
            "Confidence": r["Decision_Confidence"],
            "Purchase_Value": r["Purchase_Value"],
        })
    return pd.DataFrame(plan)

def supplier_message(row, company=""):
    if st.session_state.get("language", "English") == "Spanish":
        return (
            f"Asunto: Seguimiento urgente de supply chain — {row['SKU']} / {row['Description']}\n\n"
            f"Hola equipo de {row['Supplier']},\n\n"
            f"Estamos revisando la reposición de {row['SKU']} ({row['Description']}). "
            f"La cobertura de stock actual es de {row['Days_Cover']:.1f} días y el lead time es de {row['Lead_Time_Days']:.0f} días.\n\n"
            f"Por favor, confirma el estado actual del pedido, la fecha prevista de expedición y la fecha prevista de entrega. "
            f"Cuando corresponda, confirma también la cantidad de {row['Recommended_Order']:.0f} unidades.\n\n"
            f"Gracias,\n{company or 'Equipo de Supply Chain'}"
        )
    return (
        f"Subject: Urgent supply chain follow-up — {row['SKU']} / {row['Description']}\n\n"
        f"Hello {row['Supplier']} team,\n\n"
        f"We are reviewing replenishment for {row['SKU']} ({row['Description']}). "
        f"The current stock coverage is {row['Days_Cover']:.1f} days and the lead time is {row['Lead_Time_Days']:.0f} days.\n\n"
        f"Please confirm the current order status, expected ship date and expected delivery date. "
        f"Where applicable, please confirm the quantity of {row['Recommended_Order']:.0f} units.\n\n"
        f"Thank you,\n{company or 'Supply Chain Team'}"
    )


def _html_badge(value):
    colors = {
        "🔴 CRITICAL": "#FEE2E2", "🟠 REVIEW": "#FFEDD5",
        "🟡 EXCESS": "#FEF3C7", "🟢 OK": "#DCFCE7",
        "BUY_NOW": "#FEE2E2", "CONFIRM_PO": "#DBEAFE",
        "REVIEW": "#FFEDD5", "DO_NOT_BUY": "#FEF3C7",
        "MONITOR": "#DCFCE7", "HIGH": "#FEE2E2",
        "MEDIUM": "#FEF3C7", "LOW": "#E0E7FF",
        "WARNING": "#FFEDD5", "CRITICAL": "#FEE2E2", "OK": "#DCFCE7"
    }
    bg = colors.get(str(value), "#F3F4F6")
    return f'<span style="background:{bg};padding:4px 8px;border-radius:999px;font-weight:700;">{html_lib.escape(str(value))}</span>'

def _fmt_num(v):
    if pd.isna(v):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.1f}"
    return html_lib.escape(str(v))

def _fmt_eur(v):
    if pd.isna(v):
        return "—"
    return f"€{float(v):,.0f}"

def _bar(value, max_value, color="#2563EB"):
    width = 0 if not max_value or max_value <= 0 else min(100, max(0, float(value) / float(max_value) * 100))
    return (
        f'<div style="background:#E5E7EB;border-radius:8px;height:9px;width:100%;">'
        f'<div style="background:{color};width:{width:.1f}%;height:9px;border-radius:8px;"></div>'
        f'</div>'
    )

def _table_html(df, columns, currency_cols=None, bar_cols=None, status_cols=None):
    currency_cols = set(currency_cols or [])
    bar_cols = set(bar_cols or [])
    status_cols = set(status_cols or [])
    max_by_col = {}
    for c in bar_cols:
        if c in df.columns and len(df):
            vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
            max_by_col[c] = float(vals.max()) if len(vals) else 0

    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if c in status_cols:
                cell = _html_badge(v)
            elif c in currency_cols:
                cell = _fmt_eur(v)
            elif c in bar_cols:
                cell = f"{_fmt_num(v)}{_bar(v, max_by_col.get(c, 0))}"
            else:
                cell = _fmt_num(v)
            cells.append(f"<td>{cell}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(rows)

_REPORT_TRANSLATIONS_ES = {
    "Inventory value":"Valor de inventario", "Purchase requirement":"Necesidad de compra", "Service risk exposure":"Exposición de riesgo de servicio",
    "Excess inventory":"Exceso de inventario", "Critical":"Crítico", "Review":"Revisar", "Excess":"Exceso", "Data warnings":"Avisos de datos",
    "Latest period":"Último periodo", "Executive priorities":"Prioridades ejecutivas", "Supplier exposure":"Exposición por proveedor",
    "Weekly action plan":"Plan de acción semanal", "Data quality":"Calidad de datos", "Service risk":"Riesgo de servicio",
    "Purchase exposure":"Exposición de compras", "Inventory":"Inventario", "Priority":"Prioridad", "Timing":"Momento", "Qty":"Cantidad",
    "Purchase":"Compra", "Owner":"Responsable", "Deadline":"Fecha límite", "Reason":"Motivo", "Confidence":"Confianza",
    "Category":"Categoría", "Check":"Comprobación", "Status":"Estado", "Count":"Cantidad", "Details":"Detalles",
    "Inventory health by status":"Salud del inventario por estado", "Highest excess exposure":"Mayor exposición de exceso",
    "Coverage":"Cobertura", "Excess qty":"Cantidad en exceso", "Excess value":"Valor en exceso", "Highest service-risk exposure":"Mayor exposición de riesgo de servicio",
    "Lead time":"Lead time", "Risk qty":"Cantidad en riesgo", "Risk value":"Valor en riesgo", "Purchase lines":"Líneas de compra",
    "Units to order":"Unidades a pedir", "Suppliers":"Proveedores", "Purchase requirements by supplier":"Necesidades de compra por proveedor",
    "Lines":"Líneas", "Units":"Unidades", "Recommended purchases":"Compras recomendadas", "Unit cost":"Coste unitario",
    "Planner guidance":"Guía para el planner", "Total actions":"Total de acciones", "Immediate":"Inmediatas", "Owners":"Responsables",
    "Planner worklist":"Lista de trabajo del planner", "With critical SKUs":"Con SKUs críticos", "Supplier risk ranking":"Ranking de riesgo de proveedores",
    "Risk score":"Puntuación de riesgo", "Management interpretation":"Interpretación para dirección", "Quality status":"Estado de calidad",
    "Detailed checks":"Comprobaciones detalladas", "Immediate actions":"Acciones inmediatas", "Risk potentially addressed":"Riesgo potencialmente abordable",
    "Block replenishment":"Bloquear reposición", "Blocked exposure":"Exposición bloqueada", "Execution sequence":"Secuencia de ejecución",
    "Step":"Paso", "Task":"Tarea", "Dependency":"Dependencia", "Risk addressed":"Riesgo abordado", "Planner rationale":"Justificación para el planner",
    "Rationale":"Justificación", "Comparison":"Comparación", "Purchase delta":"Variación de compra", "Action changes":"Cambios de acción",
    "Worsened":"Empeorado", "Improved":"Mejorado", "Watch":"Vigilar", "Critical delta":"Variación de críticos", "Top changes":"Principales cambios",
    "Previous action":"Acción anterior", "Current action":"Acción actual", "Cover Δ":"Variación de cobertura", "Purchase Δ":"Variación de compra",
    "Service risk Δ":"Variación de riesgo de servicio", "Classification":"Clasificación", "Worsened / watch":"Empeorado / vigilar",
    "Structural and consistency checks for the current dataset.":"Comprobaciones estructurales y de consistencia del dataset actual.",
    "Planner-ready operational worklist with ownership and deadlines.":"Lista operativa preparada para el planner, con responsables y fechas límite.",
    "Supplier concentration, service exposure and purchasing exposure.":"Concentración de proveedores, exposición de servicio y exposición de compras.",
    "Generated":"Generado", "Decision support only. Validate purchase execution and supplier commitments before release.":"Solo para soporte a la decisión. Valida la ejecución de compras y los compromisos de los proveedores antes de su liberación.",
    "Supply Chain AI Copilot V2.0.13":"Supply Chain AI Copilot V2.0.13",
    "No comparable periods are available.":"No hay periodos comparables disponibles.",
}

def _localize_report_html(html):
    if st.session_state.get("language", "English") != "Spanish":
        return html
    for en, es in _REPORT_TRANSLATIONS_ES.items():
        html = html.replace(en, es)
    return html

def _html_shell(title, subtitle, body):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _localize_report_html(f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
:root{{--ink:#182230;--muted:#64748b;--line:#e5e7eb;--bg:#f4f7fb;--card:#ffffff;--blue:#2563eb;}}
*{{box-sizing:border-box}}
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:28px}}
.container{{max-width:1240px;margin:auto}}
.header{{background:linear-gradient(135deg,#10243d,#1d4ed8);color:white;padding:30px 34px;border-radius:20px}}
.header h1{{margin:0 0 8px;font-size:30px}}
.header p{{margin:0;color:#dbeafe}}
.meta{{margin-top:12px;font-size:12px;color:#bfdbfe}}
.section{{background:var(--card);padding:22px;margin:18px 0;border-radius:16px;box-shadow:0 4px 18px rgba(15,23,42,.06)}}
h2{{margin:0 0 16px;font-size:20px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}}
.grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:18px 0}}
.card{{background:white;border-radius:14px;padding:18px;box-shadow:0 4px 18px rgba(15,23,42,.05)}}
.label{{font-size:11px;color:#64748b;text-transform:uppercase;font-weight:800;letter-spacing:.05em}}
.value{{font-size:27px;font-weight:800;margin-top:6px}}
.sub{{font-size:12px;color:#64748b;margin-top:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#eef2f7;text-align:left;padding:11px}}
td{{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}
.note{{padding:13px 15px;border-radius:12px;background:#eff6ff;color:#1e40af}}
.warning{{padding:13px 15px;border-radius:12px;background:#fff7ed;color:#9a3412}}
.success{{padding:13px 15px;border-radius:12px;background:#f0fdf4;color:#166534}}
.small{{font-size:12px;color:#64748b}}
.footer{{margin-top:18px;font-size:11px;color:#64748b}}
@media(max-width:900px){{.grid,.grid5{{grid-template-columns:1fr 1fr}}}}
@media print{{body{{background:white;padding:10px}}.section,.card{{box-shadow:none;border:1px solid var(--line)}}.header{{break-inside:avoid}}}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>{html_lib.escape(title)}</h1>
<p>{html_lib.escape(subtitle)}</p>
<div class="meta">Generated {generated} · Supply Chain AI Copilot V2.0.13</div>
</div>
{body}
<div class="footer">Decision support only. Validate purchase execution and supplier commitments before release.</div>
</div>
</body>
</html>""")

def build_executive_html(a, raw, dq, plan):
    critical = int((a["Status"] == "🔴 CRITICAL").sum())
    review = int((a["Status"] == "🟠 REVIEW").sum())
    excess = int((a["Status"] == "🟡 EXCESS").sum())
    purchase = float(a["Purchase_Value"].sum())
    inventory = float(a["Inventory_Value"].sum())
    service_risk = float(a["Service_Risk_Value"].sum())
    excess_value = float(a["Excess_Inventory_Value"].sum())
    purchase_skus = int((a["Recommended_Order"] > 0).sum())
    top = a.sort_values("Decision_Score", ascending=False).head(10)
    supplier = a.groupby("Supplier", as_index=False).agg(
        Purchase_Value=("Purchase_Value","sum"),
        Service_Risk=("Service_Risk_Value","sum"),
        Critical=("Status", lambda s: int((s=="🔴 CRITICAL").sum())),
        Inventory=("Inventory_Value","sum"),
    ).sort_values(["Critical","Service_Risk","Purchase_Value"], ascending=[False,False,False]).head(10)
    dq_summary = data_quality_summary(raw, dq)

    body = f"""
<div class="grid">
<div class="card"><div class="label">Inventory value</div><div class="value">{_fmt_eur(inventory)}</div></div>
<div class="card"><div class="label">Purchase requirement</div><div class="value">{_fmt_eur(purchase)}</div><div class="sub">{purchase_skus} SKUs with recommendation</div></div>
<div class="card"><div class="label">Service risk exposure</div><div class="value">{_fmt_eur(service_risk)}</div></div>
<div class="card"><div class="label">Excess inventory</div><div class="value">{_fmt_eur(excess_value)}</div></div>
</div>
<div class="grid5">
<div class="card"><div class="label">Critical</div><div class="value">{critical}</div></div>
<div class="card"><div class="label">Review</div><div class="value">{review}</div></div>
<div class="card"><div class="label">Excess</div><div class="value">{excess}</div></div>
<div class="card"><div class="label">Data warnings</div><div class="value">{dq_summary["warnings"]}</div></div>
<div class="card"><div class="label">Latest period</div><div class="value" style="font-size:20px">{dq_summary["latest_period"]}</div></div>
</div>
<div class="section">
<h2>1. Executive priorities</h2>
<div class="note">The highest-priority decisions are ranked by stockout risk, economic exposure, timing and confidence.</div>
<table><thead><tr><th>SKU</th><th>Description</th><th>Supplier</th><th>Status</th><th>Action</th><th>Timing</th><th>Qty</th><th>Purchase</th></tr></thead>
<tbody>{_table_html(top, ["SKU","Description","Supplier","Status","Action","Action_Timing","Recommended_Order","Purchase_Value"], ["Purchase_Value"], status_cols=["Status","Action"])}</tbody></table>
</div>
<div class="section">
<h2>2. Supplier exposure</h2>
<table><thead><tr><th>Supplier</th><th>Critical</th><th>Service risk</th><th>Purchase exposure</th><th>Inventory</th></tr></thead>
<tbody>{_table_html(supplier, ["Supplier","Critical","Service_Risk","Purchase_Value","Inventory"], ["Service_Risk","Purchase_Value","Inventory"], bar_cols=["Purchase_Value"])}</tbody></table>
</div>
<div class="section">
<h2>3. Weekly action plan</h2>
<table><thead><tr><th>Priority</th><th>SKU</th><th>Action</th><th>Owner</th><th>Deadline</th><th>Reason</th><th>Confidence</th></tr></thead>
<tbody>{_table_html(plan.head(15), ["Priority","SKU","Action","Owner","Deadline","Reason","Confidence"], status_cols=["Action","Confidence"])}</tbody></table>
</div>
<div class="section">
<h2>4. Data quality</h2>
<p><strong>{dq_summary["ok"]}</strong> checks OK · <strong>{dq_summary["warnings"]}</strong> warnings · <strong>{dq_summary["critical"]}</strong> critical issues.</p>
<table><thead><tr><th>Category</th><th>Check</th><th>Status</th><th>Count</th><th>Details</th></tr></thead>
<tbody>{_table_html(dq, ["Category","Check","Status","Count","Details"], status_cols=["Status"])}</tbody></table>
</div>
"""
    return _html_shell("📦 Supply Chain AI — Executive Decision Report",
                       f"{dq_summary['skus']} SKUs · {dq_summary['suppliers']} suppliers · latest period {dq_summary['latest_period']}",
                       body)

def build_inventory_report_html(a, raw):
    top_excess = a.sort_values("Excess_Inventory_Value", ascending=False).head(15)
    top_service = a.sort_values("Service_Risk_Value", ascending=False).head(15)
    status = a["Status"].value_counts().rename_axis("Status").reset_index(name="SKUs")
    body = f"""
<div class="grid">
<div class="card"><div class="label">Inventory value</div><div class="value">{_fmt_eur(a["Inventory_Value"].sum())}</div></div>
<div class="card"><div class="label">Service risk</div><div class="value">{_fmt_eur(a["Service_Risk_Value"].sum())}</div></div>
<div class="card"><div class="label">Excess value</div><div class="value">{_fmt_eur(a["Excess_Inventory_Value"].sum())}</div></div>
<div class="card"><div class="label">Average cover</div><div class="value">{a["Days_Cover"].replace([np.inf,-np.inf],np.nan).mean():.1f}d</div></div>
</div>
<div class="section"><h2>Inventory health by status</h2>
<table><thead><tr><th>Status</th><th>SKUs</th></tr></thead><tbody>{_table_html(status, ["Status","SKUs"], status_cols=["Status"])}</tbody></table></div>
<div class="section"><h2>Highest excess exposure</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Supplier</th><th>Coverage</th><th>Excess qty</th><th>Excess value</th><th>Action</th></tr></thead>
<tbody>{_table_html(top_excess, ["SKU","Description","Supplier","Days_Cover","Excess_Inventory_Qty","Excess_Inventory_Value","Action"], ["Excess_Inventory_Value"], status_cols=["Action"])}</tbody></table></div>
<div class="section"><h2>Highest service-risk exposure</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Supplier</th><th>Coverage</th><th>Lead time</th><th>Risk qty</th><th>Risk value</th><th>Action</th></tr></thead>
<tbody>{_table_html(top_service, ["SKU","Description","Supplier","Days_Cover","Lead_Time_Days","Service_Risk_Qty","Service_Risk_Value","Action"], ["Service_Risk_Value"], status_cols=["Action"])}</tbody></table></div>
"""
    return _html_shell("📊 Inventory & Service Risk Report", "Inventory health, excess exposure and service-risk priorities.", body)

def build_purchase_report_html(a):
    x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
    supplier = x.groupby("Supplier", as_index=False).agg(
        Lines=("SKU","count"), Units=("Recommended_Order","sum"), Purchase_Value=("Purchase_Value","sum")
    ).sort_values("Purchase_Value", ascending=False)
    body = f"""
<div class="grid">
<div class="card"><div class="label">Purchase lines</div><div class="value">{len(x)}</div></div>
<div class="card"><div class="label">Units to order</div><div class="value">{x["Recommended_Order"].sum():,.0f}</div></div>
<div class="card"><div class="label">Purchase value</div><div class="value">{_fmt_eur(x["Purchase_Value"].sum())}</div></div>
<div class="card"><div class="label">Suppliers</div><div class="value">{x["Supplier"].nunique()}</div></div>
</div>
<div class="section"><h2>Purchase requirements by supplier</h2>
<table><thead><tr><th>Supplier</th><th>Lines</th><th>Units</th><th>Purchase value</th></tr></thead>
<tbody>{_table_html(supplier, ["Supplier","Lines","Units","Purchase_Value"], ["Purchase_Value"], bar_cols=["Purchase_Value"])}</tbody></table></div>
<div class="section"><h2>Recommended purchases</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Supplier</th><th>Qty</th><th>Unit cost</th><th>Purchase value</th><th>Coverage</th><th>Lead time</th><th>Action</th></tr></thead>
<tbody>{_table_html(x, ["SKU","Description","Supplier","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Action"], ["Purchase_Value","Unit_Cost"], status_cols=["Action"])}</tbody></table></div>
<div class="section"><h2>Planner guidance</h2><div class="note">Validate open POs and supplier ETA before releasing incremental orders. Prioritize BUY_NOW lines first.</div></div>
"""
    return _html_shell("🛒 Purchase Plan Report", "Supplier-oriented replenishment recommendations with economic exposure.", body)

def build_action_report_html(plan):
    body = f"""
<div class="grid">
<div class="card"><div class="label">Total actions</div><div class="value">{len(plan)}</div></div>
<div class="card"><div class="label">Immediate</div><div class="value">{int(plan["Deadline"].eq("Today").sum())}</div></div>
<div class="card"><div class="label">Purchase value</div><div class="value">{_fmt_eur(plan["Purchase_Value"].sum())}</div></div>
<div class="card"><div class="label">Owners</div><div class="value">{plan["Owner"].nunique()}</div></div>
</div>
<div class="section"><h2>Planner worklist</h2>
<table><thead><tr><th>Priority</th><th>SKU</th><th>Description</th><th>Supplier</th><th>Action</th><th>Owner</th><th>Deadline</th><th>Reason</th><th>Confidence</th><th>Purchase</th></tr></thead>
<tbody>{_table_html(plan, ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"], ["Purchase_Value"], status_cols=["Action","Confidence"])}</tbody></table></div>
"""
    return _html_shell("📝 Weekly Action Plan Report", "Planner-ready operational worklist with ownership and deadlines.", body)

def build_supplier_report_html(a):
    supplier = a.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU","count"),
        Critical=("Status", lambda s: int((s=="🔴 CRITICAL").sum())),
        Review=("Status", lambda s: int((s=="🟠 REVIEW").sum())),
        Purchase_Value=("Purchase_Value","sum"),
        Inventory_Value=("Inventory_Value","sum"),
        Service_Risk_Value=("Service_Risk_Value","sum"),
        Excess_Inventory_Value=("Excess_Inventory_Value","sum"),
    )
    supplier["Supplier_Risk_Score"] = (
        supplier["Critical"]*100 + supplier["Review"]*40
        + np.log1p(supplier["Service_Risk_Value"])*5
        + np.log1p(supplier["Purchase_Value"])*2
    )
    supplier = supplier.sort_values("Supplier_Risk_Score", ascending=False)
    body = f"""
<div class="grid">
<div class="card"><div class="label">Suppliers</div><div class="value">{len(supplier)}</div></div>
<div class="card"><div class="label">With critical SKUs</div><div class="value">{int((supplier["Critical"]>0).sum())}</div></div>
<div class="card"><div class="label">Service risk</div><div class="value">{_fmt_eur(supplier["Service_Risk_Value"].sum())}</div></div>
<div class="card"><div class="label">Purchase exposure</div><div class="value">{_fmt_eur(supplier["Purchase_Value"].sum())}</div></div>
</div>
<div class="section"><h2>Supplier risk ranking</h2>
<table><thead><tr><th>Supplier</th><th>SKUs</th><th>Critical</th><th>Review</th><th>Risk score</th><th>Service risk</th><th>Purchase</th><th>Inventory</th><th>Excess</th></tr></thead>
<tbody>{_table_html(supplier, ["Supplier","SKUs","Critical","Review","Supplier_Risk_Score","Service_Risk_Value","Purchase_Value","Inventory_Value","Excess_Inventory_Value"], ["Service_Risk_Value","Purchase_Value","Inventory_Value","Excess_Inventory_Value"], bar_cols=["Supplier_Risk_Score"])}</tbody></table></div>
<div class="section"><h2>Management interpretation</h2><div class="note">Supplier ranking combines critical lines, service-risk exposure and purchase exposure. It is prioritization support, not a supplier-performance scorecard.</div></div>
"""
    return _html_shell("🚚 Supplier Risk Report", "Supplier concentration, service exposure and purchasing exposure.", body)

def build_data_quality_report_html(raw, dq):
    s = data_quality_summary(raw, dq)
    status_class = "success" if s["critical"] == 0 and s["warnings"] == 0 else "warning"
    status_text = "✅ All current checks passed." if s["critical"] == 0 and s["warnings"] == 0 else "⚠️ Review the issues below before operational use."
    body = f"""
<div class="grid5">
<div class="card"><div class="label">Rows</div><div class="value">{s["rows"]}</div></div>
<div class="card"><div class="label">SKUs</div><div class="value">{s["skus"]}</div></div>
<div class="card"><div class="label">Suppliers</div><div class="value">{s["suppliers"]}</div></div>
<div class="card"><div class="label">Warnings</div><div class="value">{s["warnings"]}</div></div>
<div class="card"><div class="label">Critical</div><div class="value">{s["critical"]}</div></div>
</div>
<div class="section"><h2>Quality status</h2><div class="{status_class}">{status_text}</div></div>
<div class="section"><h2>Detailed checks</h2>
<table><thead><tr><th>Category</th><th>Check</th><th>Status</th><th>Count</th><th>Details</th></tr></thead>
<tbody>{_table_html(dq, ["Category","Check","Status","Count","Details"], status_cols=["Status"])}</tbody></table></div>
"""
    return _html_shell("🧹 Data Quality Report", "Structural and consistency checks for the current dataset.", body)



def build_planning_agent_html(planning, meta):
    body = f"""
<div class="grid">
<div class="card"><div class="label">Immediate actions</div><div class="value">{meta["immediate_count"]}</div></div>
<div class="card"><div class="label">Purchase exposure</div><div class="value">{_fmt_eur(meta["purchase_value"])}</div></div>
<div class="card"><div class="label">Service risk</div><div class="value">{_fmt_eur(meta["service_risk_value"])}</div></div>
<div class="card"><div class="label">Risk potentially addressed</div><div class="value">{_fmt_eur(meta["service_risk_addressed"])}</div></div>
</div>
<div class="grid">
<div class="card"><div class="label">BUY_NOW</div><div class="value">{meta["buy_now_count"]}</div></div>
<div class="card"><div class="label">CONFIRM_PO</div><div class="value">{meta["confirm_po_count"]}</div></div>
<div class="card"><div class="label">Block replenishment</div><div class="value">{meta["block_count"]}</div></div>
<div class="card"><div class="label">Blocked exposure</div><div class="value">{_fmt_eur(meta["blocked_purchase_exposure"])}</div></div>
</div>
<div class="section">
<h2>Execution sequence</h2>
<div class="note">Actions are ordered using urgency, service-risk exposure, economic exposure and existing decision score. This is decision support, not automatic order release.</div>
<table><thead><tr>
<th>Step</th><th>SKU</th><th>Description</th><th>Supplier</th><th>Task</th>
<th>Dependency</th><th>Timing</th><th>Qty</th><th>Purchase</th>
<th>Service risk</th><th>Risk addressed</th><th>Confidence</th>
</tr></thead>
<tbody>{_table_html(
    planning.head(20),
    ["Execution_Priority","SKU","Description","Supplier","Execution_Task","Dependency",
     "Action_Timing","Recommended_Order","Purchase_Value","Service_Risk_Value",
     "Expected_Service_Risk_Addressed","Decision_Confidence"],
    ["Purchase_Value","Service_Risk_Value","Expected_Service_Risk_Addressed"],
    status_cols=["Decision_Confidence"]
)}</tbody></table>
</div>
<div class="section">
<h2>Planner rationale</h2>
<table><thead><tr><th>SKU</th><th>Action</th><th>Rationale</th></tr></thead>
<tbody>{_table_html(planning.head(20), ["SKU","Action","Planning_Rationale"], status_cols=["Action"])}</tbody></table>
</div>
"""
    return _html_shell(
        "🧠 Supply Chain AI — Planning Agent",
        "Ordered execution sequence from inventory, demand, risk and purchasing signals.",
        body
    )

def build_change_monitor_html(comparison, meta):
    if comparison is None or comparison.empty or not meta.get("has_comparison"):
        return _html_shell(
            "🔄 Change Monitor Report",
            "No comparison available.",
            '<div class="section"><div class="note">No comparable periods are available.</div></div>'
        )
    worsened = comparison[comparison["Change_Classification"].isin(["WORSENED","WATCH"])].head(15)
    improved = comparison[comparison["Change_Classification"]=="IMPROVED"].head(15)
    body = f"""
<div class="grid">
<div class="card"><div class="label">Comparison</div><div class="value" style="font-size:20px">{meta["previous_period"]} → {meta["current_period"]}</div></div>
<div class="card"><div class="label">Purchase delta</div><div class="value">{_fmt_eur(meta["purchase_delta"])}</div></div>
<div class="card"><div class="label">Service risk delta</div><div class="value">{_fmt_eur(meta["service_risk_delta"])}</div></div>
<div class="card"><div class="label">Action changes</div><div class="value">{meta["action_changes"]}</div></div>
</div>
<div class="grid">
<div class="card"><div class="label">Worsened</div><div class="value">{meta["worsened"]}</div></div>
<div class="card"><div class="label">Improved</div><div class="value">{meta["improved"]}</div></div>
<div class="card"><div class="label">Watch</div><div class="value">{meta["watch"]}</div></div>
<div class="card"><div class="label">Critical delta</div><div class="value">{meta["critical_delta"]:+d}</div></div>
</div>
<div class="section"><h2>Top changes</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Supplier</th><th>Previous action</th><th>Current action</th><th>Cover Δ</th><th>Purchase Δ</th><th>Service risk Δ</th><th>Classification</th><th>Reason</th></tr></thead>
<tbody>{_table_html(comparison.head(20), ["SKU","Description","Supplier","Previous_Action","Current_Action","Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta","Change_Classification","Change_Reason"], ["Purchase_Value_Delta","Service_Risk_Delta"], status_cols=["Change_Classification"])}</tbody></table></div>
<div class="section"><h2>Worsened / watch</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Cover Δ</th><th>Purchase Δ</th><th>Service risk Δ</th><th>Reason</th></tr></thead>
<tbody>{_table_html(worsened, ["SKU","Description","Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta","Change_Reason"], ["Purchase_Value_Delta","Service_Risk_Delta"])}</tbody></table></div>
<div class="section"><h2>Improved</h2>
<table><thead><tr><th>SKU</th><th>Description</th><th>Cover Δ</th><th>Purchase Δ</th><th>Service risk Δ</th><th>Reason</th></tr></thead>
<tbody>{_table_html(improved, ["SKU","Description","Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta","Change_Reason"], ["Purchase_Value_Delta","Service_Risk_Delta"])}</tbody></table></div>
"""
    return _html_shell(
        "🔄 Supply Chain Change Monitor",
        "What changed between two planning periods and where the planner should look first.",
        body
    )

def build_management_pack(a, raw, dq, plan, comparison=None, comparison_meta=None):
    reports = {
        "01_Executive_Report.html": build_executive_html(a, raw, dq, plan),
        "02_Inventory_Risk_Report.html": build_inventory_report_html(a, raw),
        "03_Purchase_Plan_Report.html": build_purchase_report_html(a),
        "04_Action_Plan_Report.html": build_action_report_html(plan),
        "05_Supplier_Risk_Report.html": build_supplier_report_html(a),
        "06_Data_Quality_Report.html": build_data_quality_report_html(raw, dq),
    }
    if comparison is not None and comparison_meta is not None:
        reports["07_Change_Monitor_Report.html"] = build_change_monitor_html(comparison, comparison_meta)
    planning, planning_meta = build_planning_agent(a, comparison, comparison_meta)
    reports["08_Planning_Agent_Report.html"] = build_planning_agent_html(planning, planning_meta)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in reports.items():
            z.writestr(name, text.encode("utf-8"))
        z.writestr(
            "README.txt",
            b"Open 01_Executive_Report.html first. All reports are self-contained HTML files designed for reading, sharing and printing."
        )
    return buf.getvalue(), reports



def _xlsx_safe(v):
    if pd.isna(v):
        return ""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v

def _xlsx_write_df(ws, df, start_row, start_col, columns, workbook,
                    table_name, formats=None, widths=None, number_formats=None):
    formats = formats or {}
    widths = widths or {}
    number_formats = number_formats or {}

    header_fmt = formats.get(
        "header",
        workbook.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D",
            "border": 0, "align": "center", "valign": "vcenter", "text_wrap": True
        })
    )
    body_fmt = formats.get(
        "body",
        workbook.add_format({"border": 0, "valign": "top", "text_wrap": True})
    )

    display_columns = [(_COLUMN_TRANSLATIONS_ES.get(str(c), str(c)) if st.session_state.get("language", "English") == "Spanish" else str(c)) for c in columns]
    for j, col in enumerate(display_columns):
        ws.write(start_row, start_col + j, col, header_fmt)

    for i, row in enumerate(df[columns].itertuples(index=False, name=None), start=1):
        for j, val in enumerate(row):
            col_name = columns[j]
            fmt = body_fmt
            if col_name in number_formats:
                fmt = formats.get(col_name) or workbook.add_format({
                    "num_format": number_formats[col_name],
                    "valign": "top",
                    "text_wrap": True
                })
            ws.write(start_row + i, start_col + j, _xlsx_safe(val), fmt)

    end_row = start_row + len(df)
    end_col = start_col + len(columns) - 1
    if len(df) > 0:
        ws.add_table(start_row, start_col, end_row, end_col, {
            "name": table_name,
            "style": "Table Style Medium 2",
            "columns": [{"header": c} for c in display_columns],
        })

    for j, col in enumerate(columns):
        width = widths.get(col, 16)
        if col in {"Description", "Reason", "Details"}:
            width = max(width, 30)
        if col in {"Action", "Supplier", "SKU"}:
            width = max(width, 18)
        ws.set_column(start_col + j, start_col + j, width)

    ws.freeze_panes(start_row + 1, start_col)
    return start_row, start_col, end_row, end_col

def _xlsx_title(ws, title, subtitle, workbook, last_col=11):
    title_fmt = workbook.add_format({
        "bold": True, "font_size": 18, "font_color": "#FFFFFF",
        "bg_color": "#17365D", "valign": "vcenter"
    })
    subtitle_fmt = workbook.add_format({
        "italic": True, "font_color": "#64748B", "text_wrap": True
    })
    title = _REPORT_TRANSLATIONS_ES.get(title, tr(title)) if st.session_state.get("language", "English") == "Spanish" else title
    subtitle = _REPORT_TRANSLATIONS_ES.get(subtitle, tr(subtitle)) if st.session_state.get("language", "English") == "Spanish" else subtitle
    ws.merge_range(0, 0, 0, last_col, title, title_fmt)
    ws.merge_range(1, 0, 1, last_col, subtitle, subtitle_fmt)
    ws.set_row(0, 28)
    ws.set_row(1, 28)

def _xlsx_kpi_block(ws, workbook, row, col, width, label, value, fill):
    label_fmt = workbook.add_format({
        "bold": True, "font_color": "#334155", "bg_color": fill,
        "align": "center", "valign": "vcenter", "text_wrap": True
    })
    value_fmt = workbook.add_format({
        "bold": True, "font_size": 16, "font_color": "#0F172A",
        "bg_color": fill, "align": "center", "valign": "vcenter",
        "text_wrap": True
    })
    label = _REPORT_TRANSLATIONS_ES.get(label, tr(label)) if st.session_state.get("language", "English") == "Spanish" else label
    ws.merge_range(row, col, row, col + width - 1, label, label_fmt)
    ws.merge_range(row + 1, col, row + 2, col + width - 1, value, value_fmt)

def _xlsx_base_formats(workbook):
    return {
        "header": workbook.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D",
            "align": "center", "valign": "vcenter", "text_wrap": True
        }),
        "body": workbook.add_format({"valign": "top", "text_wrap": True}),
        "currency": workbook.add_format({"num_format": '€#,##0', "valign": "top"}),
        "number": workbook.add_format({"num_format": '#,##0.0', "valign": "top"}),
        "integer": workbook.add_format({"num_format": '#,##0', "valign": "top"}),
    }

def _xlsx_apply_basic_conditional_formats(ws, row0, row1, col0, col1, columns, workbook):
    # Status / action visual cues.
    if "Status" in columns:
        idx = columns.index("Status")
        rng = f"{xlsxwriter.utility.xl_col_to_name(col0+idx)}{row0+2}:{xlsxwriter.utility.xl_col_to_name(col0+idx)}{row1+1}"
        ws.conditional_format(rng, {"type": "text", "criteria": "containing", "value": "CRITICAL",
                                    "format": {"bg_color": "#FEE2E2", "font_color": "#991B1B"}})
        ws.conditional_format(rng, {"type": "text", "criteria": "containing", "value": "EXCESS",
                                    "format": {"bg_color": "#FEF3C7", "font_color": "#92400E"}})

def _xlsx_write_section_chart(ws, workbook, chart_type, title, categories_col, values_col, first_row, last_row, start_cell):
    if last_row < first_row:
        return
    chart = workbook.add_chart({"type": chart_type})
    if st.session_state.get("language", "English") == "Spanish":
        title = _REPORT_TRANSLATIONS_ES.get(title, tr(title))
    sheet_ref = ws.name.replace("'", "''")
    categories = f"='{sheet_ref}'!${xlsxwriter.utility.xl_col_to_name(categories_col)}${first_row+1}:${xlsxwriter.utility.xl_col_to_name(categories_col)}${last_row+1}"
    values = f"='{sheet_ref}'!${xlsxwriter.utility.xl_col_to_name(values_col)}${first_row+1}:${xlsxwriter.utility.xl_col_to_name(values_col)}${last_row+1}"
    chart.add_series({
        "name": title,
        "categories": categories,
        "values": values,
    })
    chart.set_title({"name": title})
    chart.set_style(10)
    chart.set_legend({"none": True})
    ws.insert_chart(start_cell, chart, {"x_scale": 1.05, "y_scale": 0.9})

def _xlsx_build_executive(a, raw, dq, plan):
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)

    ws = wb.add_worksheet(tr("Executive"))
    _xlsx_title(ws, "Supply Chain AI — Executive Report",
                "Dashboard view mirroring the Executive HTML report.", wb, 11)

    _xlsx_kpi_block(ws, wb, 3, 0, 3, "Inventory value", f"€{a['Inventory_Value'].sum():,.0f}", "#EAF2FF")
    _xlsx_kpi_block(ws, wb, 3, 3, 3, "Purchase requirement", f"€{a['Purchase_Value'].sum():,.0f}", "#ECFDF5")
    _xlsx_kpi_block(ws, wb, 3, 6, 3, "Service risk", f"€{a['Service_Risk_Value'].sum():,.0f}", "#FEF2F2")
    _xlsx_kpi_block(ws, wb, 3, 9, 3, "Excess exposure", f"€{a['Excess_Inventory_Value'].sum():,.0f}", "#FFF7ED")

    top = a.sort_values("Decision_Score", ascending=False).head(10)
    cols = ["SKU","Description","Supplier","Status","Action","Action_Timing",
            "Recommended_Order","Purchase_Value","Days_Cover","Lead_Time_Days"]
    row0 = 8
    _xlsx_write_df(
        ws, top, row0, 0, cols, wb, "ExecutivePriorities",
        formats={**fmt, "Purchase_Value": fmt["currency"], "Recommended_Order": fmt["integer"],
                 "Days_Cover": fmt["number"], "Lead_Time_Days": fmt["number"]},
        widths={"Description": 30, "Purchase_Value": 17, "Action": 18}
    )

    supplier = a.groupby("Supplier", as_index=False).agg(
        Critical=("Status", lambda s: int((s=="🔴 CRITICAL").sum())),
        Service_Risk=("Service_Risk_Value","sum"),
        Purchase_Value=("Purchase_Value","sum"),
        Inventory=("Inventory_Value","sum"),
    ).sort_values(["Critical","Service_Risk","Purchase_Value"], ascending=[False,False,False])
    sup_row = row0 + len(top) + 3
    _xlsx_write_df(
        ws, supplier, sup_row, 0,
        ["Supplier","Critical","Service_Risk","Purchase_Value","Inventory"],
        wb, "ExecutiveSupplierExposure",
        formats={**fmt, "Service_Risk": fmt["currency"], "Purchase_Value": fmt["currency"], "Inventory": fmt["currency"]},
        widths={"Supplier": 22}
    )
    _xlsx_write_section_chart(ws, wb, "column", "Supplier purchase exposure", 0, 3,
                              sup_row + 1, sup_row + min(len(supplier), 10), "H22")

    action_ws = wb.add_worksheet(tr("Action Plan"))
    _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
    _xlsx_write_df(
        action_ws, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "ExecutiveActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    dq_ws = wb.add_worksheet(tr("Data Quality"))
    _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecutiveDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_detailed(a, raw, dq, plan):
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)

    inv = wb.add_worksheet(tr("Inventory Risk"))
    _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
    x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
    row0 = 3
    _xlsx_write_df(
        inv, x, row0, 0,
        ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days",
         "Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
        wb, "InventoryRisk",
        formats={**fmt, "Excess_Inventory_Value": fmt["currency"], "Service_Risk_Value": fmt["currency"]},
        widths={"Description": 30}
    )
    _xlsx_write_section_chart(inv, wb, "bar", "Excess inventory value", 0, 9,
                              row0 + 1, row0 + min(len(x), 10), "M4")

    pur = wb.add_worksheet(tr("Purchase Plan"))
    _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
    x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
    row0 = 3
    _xlsx_write_df(
        pur, x, row0, 0,
        ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value",
         "Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
        wb, "PurchasePlan",
        formats={**fmt, "Unit_Cost": fmt["currency"], "Purchase_Value": fmt["currency"],
                 "Recommended_Order": fmt["integer"], "Open_PO": fmt["integer"]},
        widths={"Description": 30, "PO_Adequacy": 16}
    )
    _xlsx_write_section_chart(pur, wb, "column", "Purchase value by SKU", 0, 6,
                              row0 + 1, row0 + min(len(x), 10), "M4")

    act = wb.add_worksheet(tr("Action Plan"))
    _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
    _xlsx_write_df(
        act, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "DetailedActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    sup = wb.add_worksheet(tr("Supplier Risk"))
    _xlsx_title(sup, "Supplier Risk", "Risk concentration and economic exposure.", wb, 8)
    s = a.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU","count"),
        Critical=("Status", lambda z: int((z=="🔴 CRITICAL").sum())),
        Review=("Status", lambda z: int((z=="🟠 REVIEW").sum())),
        Purchase_Value=("Purchase_Value","sum"),
        Inventory_Value=("Inventory_Value","sum"),
        Service_Risk_Value=("Service_Risk_Value","sum"),
        Excess_Inventory_Value=("Excess_Inventory_Value","sum"),
    )
    s["Supplier_Risk_Score"] = s["Critical"]*100 + s["Review"]*40 + np.log1p(s["Service_Risk_Value"])*5 + np.log1p(s["Purchase_Value"])*2
    s = s.sort_values("Supplier_Risk_Score", ascending=False)
    row0 = 3
    _xlsx_write_df(
        sup, s, row0, 0,
        ["Supplier","SKUs","Critical","Review","Supplier_Risk_Score","Service_Risk_Value",
         "Purchase_Value","Inventory_Value","Excess_Inventory_Value"],
        wb, "SupplierRisk",
        formats={**fmt, "Service_Risk_Value": fmt["currency"], "Purchase_Value": fmt["currency"],
                 "Inventory_Value": fmt["currency"], "Excess_Inventory_Value": fmt["currency"],
                 "Supplier_Risk_Score": fmt["number"]},
        widths={"Supplier": 22}
    )
    _xlsx_write_section_chart(sup, wb, "column", "Supplier risk score", 0, 4,
                              row0 + 1, row0 + min(len(s), 10), "K4")

    dq_ws = wb.add_worksheet(tr("Data Quality"))
    _xlsx_title(dq_ws, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_complete(a, raw, dq, plan):
    wb = _xlsx_build_detailed(a, raw, dq, plan)

    # Complete pack adds Executive + Source Data to the detailed workbook.
    ex = wb.add_worksheet(tr("Executive"))
    _xlsx_title(ex, "Supply Chain AI — Complete Management Pack",
                "Executive dashboard for the full workbook.", wb, 11)
    _xlsx_kpi_block(ex, wb, 3, 0, 3, "Inventory value", f"€{a['Inventory_Value'].sum():,.0f}", "#EAF2FF")
    _xlsx_kpi_block(ex, wb, 3, 3, 3, "Purchase requirement", f"€{a['Purchase_Value'].sum():,.0f}", "#ECFDF5")
    _xlsx_kpi_block(ex, wb, 3, 6, 3, "Service risk", f"€{a['Service_Risk_Value'].sum():,.0f}", "#FEF2F2")
    _xlsx_kpi_block(ex, wb, 3, 9, 3, "Excess exposure", f"€{a['Excess_Inventory_Value'].sum():,.0f}", "#FFF7ED")
    top = a.sort_values("Decision_Score", ascending=False).head(10)
    _xlsx_write_df(
        ex, top, 8, 0,
        ["SKU","Description","Supplier","Status","Action","Action_Timing","Recommended_Order","Purchase_Value"],
        wb, "CompleteExecutivePriorities",
        formats={**_xlsx_base_formats(wb), "Purchase_Value": wb.add_format({"num_format": '€#,##0'})},
        widths={"Description": 30}
    )

    src_ws = wb.add_worksheet(tr("Source Data"))
    _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
    _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "SourceData")

    return wb



def _excel_planning_agent_bytes(planning, meta):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)
    ws = wb.add_worksheet(tr("Planning Agent"))

    _xlsx_title(
        ws,
        "Supply Chain AI — Planning Agent",
        "Ordered execution sequence generated from the decision engine.",
        wb, 11
    )
    _xlsx_kpi_block(ws, wb, 3, 0, 3, "Immediate actions", str(meta["immediate_count"]), "#FEF3C7")
    _xlsx_kpi_block(ws, wb, 3, 3, 3, "Purchase exposure", f"€{meta['purchase_value']:,.0f}", "#ECFDF5")
    _xlsx_kpi_block(ws, wb, 3, 6, 3, "Service risk", f"€{meta['service_risk_value']:,.0f}", "#FEF2F2")
    _xlsx_kpi_block(ws, wb, 3, 9, 3, "Risk addressed", f"€{meta['service_risk_addressed']:,.0f}", "#EAF2FF")

    cols = [
        "Execution_Priority","SKU","Description","Supplier","Action","Execution_Task",
        "Dependency","Action_Timing","Recommended_Order","Purchase_Value",
        "Service_Risk_Value","Expected_Service_Risk_Addressed","Blocked_Purchase_Exposure",
        "Decision_Confidence","Planning_Rationale"
    ]
    _xlsx_write_df(
        ws, planning.head(30), 8, 0, cols, wb, "PlanningAgentQueue",
        formats={
            **fmt,
            "Purchase_Value": fmt["currency"],
            "Service_Risk_Value": fmt["currency"],
            "Expected_Service_Risk_Addressed": fmt["currency"],
            "Blocked_Purchase_Exposure": fmt["currency"],
            "Recommended_Order": fmt["integer"],
        },
        widths={"Description":30,"Execution_Task":28,"Dependency":30,"Planning_Rationale":48}
    )
    action_counts = planning["Action"].value_counts().reset_index()
    action_counts.columns = ["Action","SKUs"]
    base_row = 42
    summary_fmt = wb.add_format({"bold":True,"font_color":"#FFFFFF","bg_color":"#17365D","align":"center"})
    ws.write_row(base_row, 0, ["Action","SKUs"], summary_fmt)
    for i, r in action_counts.iterrows():
        ws.write(base_row+1+i, 0, r["Action"])
        ws.write(base_row+1+i, 1, int(r["SKUs"]))
    chart = wb.add_chart({"type":"column"})
    chart.add_series({
        "name":"SKUs",
        "categories":f"='Planning Agent'!$A${base_row+2}:$A${base_row+1+len(action_counts)}",
        "values":f"='Planning Agent'!$B${base_row+2}:$B${base_row+1+len(action_counts)}"
    })
    chart.set_title({"name":"Execution workload"})
    chart.set_legend({"none":True})
    chart.set_style(10)
    ws.insert_chart("R9", chart, {"x_scale":1.0,"y_scale":0.9})
    ws.freeze_panes(9,0)
    wb.close()
    return buf.getvalue()

def _excel_change_monitor_bytes(comparison, comparison_meta):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    if comparison is None or comparison.empty or not comparison_meta.get("has_comparison"):
        raise ValueError("No comparable periods are available.")

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)
    ws = wb.add_worksheet(tr("Change Monitor"))

    _xlsx_title(
        ws,
        "Supply Chain AI — Change Monitor",
        f"{comparison_meta['previous_period']} → {comparison_meta['current_period']} · period-over-period decision changes",
        wb, 10
    )

    _xlsx_kpi_block(ws, wb, 3, 0, 3, "Purchase delta",
                    f"€{comparison_meta['purchase_delta']:,.0f}", "#ECFDF5")
    _xlsx_kpi_block(ws, wb, 3, 3, 3, "Service risk delta",
                    f"€{comparison_meta['service_risk_delta']:,.0f}", "#FEF2F2")
    _xlsx_kpi_block(ws, wb, 3, 6, 3, "Worsened",
                    str(comparison_meta["worsened"]), "#FEF3C7")
    _xlsx_kpi_block(ws, wb, 3, 9, 3, "Action changes",
                    str(comparison_meta["action_changes"]), "#EAF2FF")

    ws.write(6, 0, "Interpretation", wb.add_format({
        "bold": True, "font_color": "#17365D", "font_size": 12
    }))
    ws.merge_range(
        6, 1, 6, 10,
        "Focus first on WORSENED and WATCH lines; the table explains the main operational deltas.",
        wb.add_format({"font_color": "#475569", "text_wrap": True})
    )

    columns = [
        "Priority","SKU","Description","Supplier",
        "Previous_Action","Current_Action","Action_Transition",
        "Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
        "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta",
        "Sales_Delta_Pct","Change_Classification","Change_Reason"
    ]
    cm = comparison.copy()
    if "Priority" not in cm.columns:
        cm["Priority"] = np.arange(1, len(cm) + 1)
    cm = cm[columns].head(30)

    _xlsx_write_df(
        ws, cm, 8, 0, columns, wb, "ChangeMonitorExport",
        formats={
            **fmt,
            "Purchase_Value_Delta": fmt["currency"],
            "Service_Risk_Delta": fmt["currency"],
            "Excess_Value_Delta": fmt["currency"],
            "Days_Cover_Delta": fmt["number"],
            "Sales_Delta_Pct": fmt["number"],
        },
        widths={
            "Description": 30,
            "Action_Transition": 28,
            "Change_Reason": 42,
            "Previous_Action": 18,
            "Current_Action": 18,
            "Change_Classification": 18,
        }
    )

    class_counts = (
        comparison["Change_Classification"]
        .value_counts()
        .reindex(["WORSENED", "WATCH", "IMPROVED", "STABLE"], fill_value=0)
        .reset_index()
    )
    class_counts.columns = ["Classification", "SKUs"]
    summary_row = 41
    summary_header = wb.add_format({
        "bold": True, "font_color": "#FFFFFF",
        "bg_color": "#17365D", "align": "center"
    })
    ws.write_row(summary_row, 0, ["Classification", "SKUs"], summary_header)
    for i, row in class_counts.iterrows():
        ws.write(summary_row + 1 + i, 0, row["Classification"])
        ws.write(summary_row + 1 + i, 1, int(row["SKUs"]))

    chart = wb.add_chart({"type": "column"})
    chart.add_series({
        "name": "SKUs",
        "categories": f"='Change Monitor'!$A${summary_row+2}:$A${summary_row+1+len(class_counts)}",
        "values": f"='Change Monitor'!$B${summary_row+2}:$B${summary_row+1+len(class_counts)}",
    })
    chart.set_title({"name": "Change classification"})
    chart.set_legend({"none": True})
    chart.set_style(10)
    ws.insert_chart("R9", chart, {"x_scale": 1.0, "y_scale": 0.9})
    ws.freeze_panes(9, 0)

    wb.close()
    return buf.getvalue()

def _excel_export_bytes(kind, a, raw, dq, plan, comparison=None, comparison_meta=None):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    wb = {"executive": _xlsx_build_executive, "detailed": _xlsx_build_detailed, "complete": _xlsx_build_complete}[kind](a, raw, dq, plan)
    # Workbook is already in memory; close it by accessing its underlying
    # buffer is not exposed, so rebuild using a helper that returns bytes.
    # To keep this deterministic, generate through a common bytes wrapper below.
    raise RuntimeError("Internal workbook wrapper not initialized.")

def _excel_bytes(kind, a, raw, dq, plan):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    # Build directly so workbook.close() flushes the BytesIO.
    buf = io.BytesIO()
    if kind == "executive":
        _xlsx_rebuild = _xlsx_build_executive
    elif kind == "detailed":
        _xlsx_rebuild = _xlsx_build_detailed
    else:
        _xlsx_rebuild = _xlsx_build_complete

    # Patch the builders to accept a file-like target by temporarily
    # serializing sheet content is more complex; instead replicate by using
    # xlsxwriter's constructor in these wrapper builders.
    # The builders above need a stream. We'll use the deterministic builder
    # below for all three.
    return _xlsx_stream_build(kind, a, raw, dq, plan, comparison, comparison_meta)

def _xlsx_stream_build(kind, a, raw, dq, plan, comparison=None, comparison_meta=None):
    buf = io.BytesIO()
    # Builders create workbook instances; to ensure close flushes to buf,
    # we use a dedicated local writer around a precomputed workbook layout.
    # Implement with temporary file-like workbook by dispatching the layout
    # routines that accept workbook objects.
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})

    if kind == "executive":
        fmt = _xlsx_base_formats(wb)
        ws = wb.add_worksheet(tr("Executive"))
        _xlsx_title(ws, "Supply Chain AI — Executive Report", "Dashboard view mirroring the Executive HTML report.", wb, 11)
        _xlsx_kpi_block(ws, wb, 3, 0, 3, "Inventory value", f"€{a['Inventory_Value'].sum():,.0f}", "#EAF2FF")
        _xlsx_kpi_block(ws, wb, 3, 3, 3, "Purchase requirement", f"€{a['Purchase_Value'].sum():,.0f}", "#ECFDF5")
        _xlsx_kpi_block(ws, wb, 3, 6, 3, "Service risk", f"€{a['Service_Risk_Value'].sum():,.0f}", "#FEF2F2")
        _xlsx_kpi_block(ws, wb, 3, 9, 3, "Excess exposure", f"€{a['Excess_Inventory_Value'].sum():,.0f}", "#FFF7ED")
        top = a.sort_values("Decision_Score", ascending=False).head(10)
        _xlsx_write_df(ws, top, 8, 0,
                       ["SKU","Description","Supplier","Status","Action","Action_Timing","Recommended_Order","Purchase_Value","Days_Cover","Lead_Time_Days"],
                       wb, "ExecPriorities", formats={**fmt, "Purchase_Value": fmt["currency"], "Recommended_Order": fmt["integer"], "Days_Cover": fmt["number"], "Lead_Time_Days": fmt["number"]},
                       widths={"Description": 30})
        action_ws = wb.add_worksheet(tr("Action Plan"))
        _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
        _xlsx_write_df(action_ws, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "ExecActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        dq_ws = wb.add_worksheet(tr("Data Quality"))
        _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
        _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecDQ", widths={"Check":30,"Details":42})

    elif kind == "detailed":
        fmt = _xlsx_base_formats(wb)
        inv = wb.add_worksheet(tr("Inventory Risk"))
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "InventoryRisk", formats={**fmt, "Excess_Inventory_Value": fmt["currency"], "Service_Risk_Value": fmt["currency"]}, widths={"Description":30})
        _xlsx_write_section_chart(inv, wb, "bar", "Excess inventory value", 0, 9, 4, min(3+len(x),13), "M4")
        pur = wb.add_worksheet(tr("Purchase Plan"))
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "PurchasePlan", formats={**fmt, "Unit_Cost": fmt["currency"], "Purchase_Value": fmt["currency"], "Recommended_Order":fmt["integer"], "Open_PO":fmt["integer"]}, widths={"Description":30})
        _xlsx_write_section_chart(pur, wb, "column", "Purchase value by SKU", 0, 6, 4, min(3+len(x),13), "M4")
        act = wb.add_worksheet(tr("Action Plan"))
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "DetailedActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        sup = wb.add_worksheet(tr("Supplier Risk"))
        _xlsx_title(sup, "Supplier Risk", "Risk concentration and economic exposure.", wb, 8)
        s = a.groupby("Supplier", as_index=False).agg(
            SKUs=("SKU","count"), Critical=("Status", lambda z:int((z=="🔴 CRITICAL").sum())),
            Review=("Status", lambda z:int((z=="🟠 REVIEW").sum())),
            Purchase_Value=("Purchase_Value","sum"), Inventory_Value=("Inventory_Value","sum"),
            Service_Risk_Value=("Service_Risk_Value","sum"), Excess_Inventory_Value=("Excess_Inventory_Value","sum")
        )
        s["Supplier_Risk_Score"] = s["Critical"]*100 + s["Review"]*40 + np.log1p(s["Service_Risk_Value"])*5 + np.log1p(s["Purchase_Value"])*2
        s = s.sort_values("Supplier_Risk_Score", ascending=False)
        _xlsx_write_df(sup, s, 3, 0,
                       ["Supplier","SKUs","Critical","Review","Supplier_Risk_Score","Service_Risk_Value","Purchase_Value","Inventory_Value","Excess_Inventory_Value"],
                       wb, "SupplierRisk", formats={**fmt, "Service_Risk_Value":fmt["currency"],"Purchase_Value":fmt["currency"],"Inventory_Value":fmt["currency"],"Excess_Inventory_Value":fmt["currency"]}, widths={"Supplier":22})
        _xlsx_write_section_chart(sup, wb, "column", "Supplier risk score", 0, 4, 4, min(3+len(s),13), "K4")
        dqs = wb.add_worksheet(tr("Data Quality"))
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDQ", widths={"Check":30,"Details":42})

    else:
        fmt = _xlsx_base_formats(wb)
        ex = wb.add_worksheet(tr("Executive"))
        _xlsx_title(ex, "Supply Chain AI — Complete Management Pack", "Executive dashboard for the full workbook.", wb, 11)
        _xlsx_kpi_block(ex, wb, 3, 0, 3, "Inventory value", f"€{a['Inventory_Value'].sum():,.0f}", "#EAF2FF")
        _xlsx_kpi_block(ex, wb, 3, 3, 3, "Purchase requirement", f"€{a['Purchase_Value'].sum():,.0f}", "#ECFDF5")
        _xlsx_kpi_block(ex, wb, 3, 6, 3, "Service risk", f"€{a['Service_Risk_Value'].sum():,.0f}", "#FEF2F2")
        _xlsx_kpi_block(ex, wb, 3, 9, 3, "Excess exposure", f"€{a['Excess_Inventory_Value'].sum():,.0f}", "#FFF7ED")
        top = a.sort_values("Decision_Score", ascending=False).head(10)
        _xlsx_write_df(ex, top, 8, 0,
                       ["SKU","Description","Supplier","Status","Action","Action_Timing","Recommended_Order","Purchase_Value"],
                       wb, "CompleteExec", formats={**fmt, "Purchase_Value":fmt["currency"],"Recommended_Order":fmt["integer"]}, widths={"Description":30})

        # Detailed sheets are included as part of the complete pack.
        inv = wb.add_worksheet(tr("Inventory Risk"))
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "CompleteInventoryRisk", formats={**fmt,"Excess_Inventory_Value":fmt["currency"],"Service_Risk_Value":fmt["currency"]}, widths={"Description":30})

        pur = wb.add_worksheet(tr("Purchase Plan"))
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "CompletePurchasePlan", formats={**fmt,"Unit_Cost":fmt["currency"],"Purchase_Value":fmt["currency"],"Recommended_Order":fmt["integer"],"Open_PO":fmt["integer"]}, widths={"Description":30})

        act = wb.add_worksheet(tr("Action Plan"))
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "CompleteActionPlan", formats={**fmt,"Purchase_Value":fmt["currency"]}, widths={"Description":30,"Reason":34})

        sup = wb.add_worksheet(tr("Supplier Risk"))
        _xlsx_title(sup, "Supplier Risk", "Risk concentration and economic exposure.", wb, 8)
        s = a.groupby("Supplier", as_index=False).agg(
            SKUs=("SKU","count"), Critical=("Status", lambda z:int((z=="🔴 CRITICAL").sum())),
            Review=("Status", lambda z:int((z=="🟠 REVIEW").sum())),
            Purchase_Value=("Purchase_Value","sum"), Inventory_Value=("Inventory_Value","sum"),
            Service_Risk_Value=("Service_Risk_Value","sum"), Excess_Inventory_Value=("Excess_Inventory_Value","sum")
        )
        s["Supplier_Risk_Score"] = s["Critical"]*100 + s["Review"]*40 + np.log1p(s["Service_Risk_Value"])*5 + np.log1p(s["Purchase_Value"])*2
        s = s.sort_values("Supplier_Risk_Score", ascending=False)
        _xlsx_write_df(sup, s, 3, 0,
                       ["Supplier","SKUs","Critical","Review","Supplier_Risk_Score","Service_Risk_Value","Purchase_Value","Inventory_Value","Excess_Inventory_Value"],
                       wb, "CompleteSupplierRisk", formats={**fmt,"Service_Risk_Value":fmt["currency"],"Purchase_Value":fmt["currency"],"Inventory_Value":fmt["currency"],"Excess_Inventory_Value":fmt["currency"]}, widths={"Supplier":22})

        dqs = wb.add_worksheet(tr("Data Quality"))
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "CompleteDQ", widths={"Check":30,"Details":42})

        src_ws = wb.add_worksheet(tr("Source Data"))
        _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
        _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "CompleteSourceData")


    if comparison is not None and comparison_meta is not None and not comparison.empty:
        cm = wb.add_worksheet(tr("Change Monitor"))
        _xlsx_title(
            cm,
            "Change Monitor",
            f"{comparison_meta['previous_period']} → {comparison_meta['current_period']} · period-over-period changes",
            wb, 10
        )
        _xlsx_kpi_block(cm, wb, 3, 0, 3, "Purchase delta", f"€{comparison_meta['purchase_delta']:,.0f}", "#ECFDF5")
        _xlsx_kpi_block(cm, wb, 3, 3, 3, "Service risk delta", f"€{comparison_meta['service_risk_delta']:,.0f}", "#FEF2F2")
        _xlsx_kpi_block(cm, wb, 3, 6, 3, "Worsened", str(comparison_meta["worsened"]), "#FEF3C7")
        _xlsx_kpi_block(cm, wb, 3, 9, 3, "Action changes", str(comparison_meta["action_changes"]), "#EAF2FF")
        cm_df = comparison.head(20)
        fmt_cm = _xlsx_base_formats(wb)
        _xlsx_write_df(
            cm, cm_df, 8, 0,
            ["SKU","Description","Supplier","Previous_Action","Current_Action","Action_Transition",
             "Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
             "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta",
             "Sales_Delta_Pct","Change_Classification","Change_Reason"],
            wb, "ChangeMonitor",
            formats={
                **fmt_cm,
                "Purchase_Value_Delta": fmt_cm["currency"],
                "Service_Risk_Delta": fmt_cm["currency"],
                "Excess_Value_Delta": fmt_cm["currency"],
                "Days_Cover_Delta": fmt_cm["number"],
                "Sales_Delta_Pct": fmt_cm["number"],
            },
            widths={"Description":28,"Action_Transition":28,"Change_Reason":40}
        )
        try:
            _xlsx_write_section_chart(
                cm, wb, "column", "Service risk delta by SKU", 0, 10,
                9, 9 + min(len(cm_df), 10), "Q9"
            )
        except Exception:
            pass

    planning_sheet_df, planning_meta_xlsx = build_planning_agent(a, comparison, comparison_meta)
    pa = wb.add_worksheet(tr("Planning Agent"))
    fmt_pa = _xlsx_base_formats(wb)
    _xlsx_title(pa, "Supply Chain AI — Planning Agent",
                "Ordered execution sequence generated by the decision engine.", wb, 11)
    _xlsx_kpi_block(pa, wb, 3, 0, 3, "Immediate actions", str(planning_meta_xlsx["immediate_count"]), "#FEF3C7")
    _xlsx_kpi_block(pa, wb, 3, 3, 3, "Purchase exposure", f"€{planning_meta_xlsx['purchase_value']:,.0f}", "#ECFDF5")
    _xlsx_kpi_block(pa, wb, 3, 6, 3, "Service risk", f"€{planning_meta_xlsx['service_risk_value']:,.0f}", "#FEF2F2")
    _xlsx_kpi_block(pa, wb, 3, 9, 3, "Risk addressed", f"€{planning_meta_xlsx['service_risk_addressed']:,.0f}", "#EAF2FF")

    pa_cols = [
        "Execution_Priority","SKU","Description","Supplier","Action","Execution_Task",
        "Dependency","Action_Timing","Recommended_Order","Purchase_Value",
        "Service_Risk_Value","Expected_Service_Risk_Addressed","Blocked_Purchase_Exposure",
        "Decision_Confidence","Planning_Rationale"
    ]
    _xlsx_write_df(
        pa, planning_sheet_df.head(30), 8, 0, pa_cols, wb, "PlanningAgentQueue",
        formats={
            **fmt_pa,
            "Purchase_Value": fmt_pa["currency"],
            "Service_Risk_Value": fmt_pa["currency"],
            "Expected_Service_Risk_Addressed": fmt_pa["currency"],
            "Blocked_Purchase_Exposure": fmt_pa["currency"],
            "Recommended_Order": fmt_pa["integer"],
        },
        widths={"Description":30,"Execution_Task":28,"Dependency":30,"Planning_Rationale":48}
    )
    pa.freeze_panes(9, 0)

    wb.close()
    return buf.getvalue()

def _excel_export_bytes(kind, a, raw, dq, plan, comparison=None, comparison_meta=None):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    return _xlsx_stream_build(kind, a, raw, dq, plan, comparison, comparison_meta)


def validate(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return False, f"Faltan columnas obligatorias: {', '.join(missing)}"
    return True, ""

def analyze(df, safety_days=10, service_level=0.95):
    df = df.copy()
    for c in ["Sales","Stock","Open_PO","Lead_Time_Days","MOQ","Unit_Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Basic demand statistics
    g = df.sort_values(["Year","Month"]).groupby(
        ["SKU","Description","Supplier","Lead_Time_Days","MOQ","Unit_Cost"],
        as_index=False
    )
    a = g.agg(Annual_Sales=("Sales","sum"), Stock=("Stock","last"), Open_PO=("Open_PO","last"))

    a["Avg_Monthly_Demand"] = a["Annual_Sales"] / 12
    a["Avg_Daily_Demand"] = a["Annual_Sales"] / 365
    a["Lead_Time_Demand"] = a["Avg_Daily_Demand"] * a["Lead_Time_Days"]

    # Robust but transparent MVP safety stock using demand CV.
    cv_map, forecast_map, trend_map = {}, {}, {}
    for sku, x in df.groupby("SKU"):
        x = x.sort_values(["Year","Month"])
        vals = x["Sales"].tail(12).to_numpy(dtype=float)
        mean = vals.mean() if len(vals) else 0
        std = vals.std(ddof=0) if len(vals) else 0
        cv = std / mean if mean else 0
        cv_map[sku] = cv

        recent = vals[-6:] if len(vals) >= 6 else vals
        w = np.arange(1, len(recent)+1)
        weighted = float(np.average(recent, weights=w)) if len(recent) else 0
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 3 else 0
        forecast = max(0, weighted + slope)
        forecast_map[sku] = forecast
        trend_map[sku] = slope

    # Translate desired service level into conservative z values.
    z_map = {0.90: 1.28, 0.95: 1.65, 0.975: 1.96, 0.99: 2.33}
    z = z_map.get(round(service_level, 3), 1.65)

    a["Demand_CV"] = a["SKU"].map(cv_map).fillna(0)
    a["Forecast_Next_Month"] = a["SKU"].map(forecast_map).fillna(0)
    a["Trend_Units_Per_Month"] = a["SKU"].map(trend_map).fillna(0)
    a["Forecast_Change_Pct"] = np.where(
        a["Avg_Monthly_Demand"] > 0,
        (a["Forecast_Next_Month"] / a["Avg_Monthly_Demand"] - 1) * 100,
        0
    )

    # A transparent approximation: monthly demand std -> daily uncertainty.
    a["Daily_Demand_Std"] = a["Avg_Daily_Demand"] * a["Demand_CV"]
    a["Safety_Stock"] = (z * a["Daily_Demand_Std"] * np.sqrt(a["Lead_Time_Days"])).clip(lower=0)
    # Respect an operational floor set by planner.
    a["Safety_Stock"] = np.maximum(a["Safety_Stock"], a["Avg_Daily_Demand"] * safety_days)

    a["Required_Stock"] = a["Lead_Time_Demand"] + a["Safety_Stock"]
    a["Days_Cover"] = a.apply(
        lambda r: r["Stock"] / r["Avg_Daily_Demand"] if r["Avg_Daily_Demand"] > 0 else np.inf,
        axis=1
    )
    a["Net_Available"] = a["Stock"] + a["Open_PO"]
    a["Inventory_Value"] = a["Stock"] * a["Unit_Cost"]
    a["Raw_Order_Qty"] = (a["Required_Stock"] - a["Net_Available"]).clip(lower=0)
    a["Recommended_Order"] = [
        math.ceil(q/m)*m if q > 0 and m > 0 else 0
        for q, m in zip(a["Raw_Order_Qty"], a["MOQ"])
    ]
    a["Purchase_Value"] = a["Recommended_Order"] * a["Unit_Cost"]

    a["Open_PO_Cover_Days"] = a.apply(
        lambda r: r["Open_PO"] / r["Avg_Daily_Demand"] if r["Avg_Daily_Demand"] > 0 else np.inf,
        axis=1
    )
    a["Projected_Stock_After_PO"] = a["Stock"] + a["Open_PO"]
    a["Projected_Days_After_PO"] = a.apply(
        lambda r: r["Projected_Stock_After_PO"] / r["Avg_Daily_Demand"]
        if r["Avg_Daily_Demand"] > 0 else np.inf, axis=1
    )
    a["Service_Risk_Qty"] = (a["Lead_Time_Demand"] - a["Stock"]).clip(lower=0)
    a["Service_Risk_Value"] = a["Service_Risk_Qty"] * a["Unit_Cost"]
    a["Excess_Inventory_Qty"] = (a["Stock"] - a["Required_Stock"]).clip(lower=0)
    a["Excess_Inventory_Value"] = a["Excess_Inventory_Qty"] * a["Unit_Cost"]
    a["PO_Adequacy"] = np.where(
        a["Open_PO"] >= a["Required_Stock"] - a["Stock"], "ADEQUATE",
        np.where(a["Open_PO"] > 0, "PARTIAL", "NONE")
    )

    # Action timing / urgency
    a["Days_To_Stockout"] = a["Days_Cover"]
    a["Lead_Time_Gap_Days"] = a["Days_Cover"] - a["Lead_Time_Days"]
    a["Stockout_Buffer_Days"] = a["Days_Cover"] - a["Lead_Time_Days"]

    def status(r):
        if r["Stock"] < r["Lead_Time_Demand"]:
            return "🔴 CRITICAL"
        if r["Stock"] < r["Required_Stock"]:
            return "🟠 REVIEW"
        if r["Days_Cover"] > r["Lead_Time_Days"] + safety_days*3:
            return "🟡 EXCESS"
        return "🟢 OK"

    # Status MUST be created before any function reads r["Status"].
    a["Status"] = a.apply(status, axis=1)

    def action(r):
        if r["Status"] == "🔴 CRITICAL":
            return "BUY_NOW"
        if r["Status"] == "🟠 REVIEW":
            # If an open PO exists, the operational action is to validate it
            # before ordering another quantity.
            if r["Open_PO"] > 0:
                return "CONFIRM_PO"
            if r["Days_Cover"] <= r["Lead_Time_Days"] + 7:
                return "BUY_NOW"
            return "REVIEW"
        if r["Status"] == "🟡 EXCESS":
            return "DO_NOT_BUY"
        return "MONITOR"

    a["Action"] = a.apply(action, axis=1)

    def timing(r):
        if r["Action"] == "BUY_NOW":
            return "Immediate"
        if r["Action"] == "CONFIRM_PO":
            return "This week"
        if r["Action"] == "REVIEW":
            return "Next planning cycle"
        if r["Action"] == "DO_NOT_BUY":
            return "Block replenishment"
        return "Monitor"

    def confidence(r):
        if r["Annual_Sales"] <= 0:
            return "LOW"
        if r["Demand_CV"] <= 0.25:
            return "HIGH"
        if r["Demand_CV"] <= 0.50:
            return "MEDIUM"
        return "LOW"

    a["Action_Timing"] = a.apply(timing, axis=1)
    a["Decision_Confidence"] = a.apply(confidence, axis=1)

    # ABC
    a["Annual_Consumption_Value"] = a["Annual_Sales"] * a["Unit_Cost"]
    a = a.sort_values("Annual_Consumption_Value", ascending=False).reset_index(drop=True)
    total = a["Annual_Consumption_Value"].sum()
    a["Cum_Value_Pct"] = a["Annual_Consumption_Value"].cumsum()/total*100 if total else 0
    a["ABC"] = a["Cum_Value_Pct"].apply(lambda x: "A" if x <= 80 else ("B" if x <= 95 else "C"))
    a["XYZ"] = a["Demand_CV"].apply(lambda x: "X" if x <= .25 else ("Y" if x <= .50 else "Z"))
    a["ABC_XYZ"] = a["ABC"] + a["XYZ"]

    # Decision score
    a["Risk_Score"] = a["Status"].map({"🔴 CRITICAL":100,"🟠 REVIEW":60,"🟡 EXCESS":20,"🟢 OK":0}).fillna(0)
    a["Decision_Score"] = a["Risk_Score"] + np.log1p(a["Purchase_Value"] + a["Inventory_Value"]) * 5 + np.maximum(a["Trend_Units_Per_Month"], 0) * .01

    # Simple data quality signals
    a["Data_Quality"] = np.where(a["Annual_Sales"] <= 0, "⚠️ No demand", "OK")
    return a

def build_logistics_dashboard(a, raw):
    """Build the KPI layer used by the Logistics Dashboard and its exports."""
    x = a.copy()
    r = raw.copy()

    for c in ["Sales", "Stock", "Open_PO", "Lead_Time_Days", "Unit_Cost"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    # Use a true trailing-12-month sales base when the source contains more than 12 months.
    # Inventory turns are calculated against average monthly inventory value over the same period.
    annual_sales_value = float((x["Annual_Sales"] * x["Unit_Cost"]).sum())
    avg_inventory_value_12m = inventory_value = float(x["Inventory_Value"].sum())
    if all(c in r.columns for c in ["Year", "Month"]):
        r["_PeriodKey"] = pd.to_numeric(r["Year"], errors="coerce").fillna(0).astype(int) * 100 + pd.to_numeric(r["Month"], errors="coerce").fillna(0).astype(int)
        periods = sorted(r["_PeriodKey"].unique())
        recent_periods = periods[-12:]
        recent = r[r["_PeriodKey"].isin(recent_periods)].copy()
        if not recent.empty:
            recent["_SalesValue"] = recent["Sales"] * recent["Unit_Cost"]
            recent["_InventoryValue"] = recent["Stock"] * recent["Unit_Cost"]
            annual_sales_value = float(recent["_SalesValue"].sum())
            monthly_inventory = recent.groupby("_PeriodKey")["_InventoryValue"].sum()
            avg_inventory_value_12m = float(monthly_inventory.mean()) if not monthly_inventory.empty else inventory_value
    inventory_value = float(x["Inventory_Value"].sum())
    open_po_value = float((x["Open_PO"] * x["Unit_Cost"]).sum())
    purchase_value = float(x["Purchase_Value"].sum())
    excess_value = float(x["Excess_Inventory_Value"].sum())
    service_risk_value = float(x["Service_Risk_Value"].sum())

    total_avg_daily_demand = float(x["Avg_Daily_Demand"].sum())
    aggregate_cover = float(x["Stock"].sum() / total_avg_daily_demand) if total_avg_daily_demand > 0 else np.nan
    inventory_turns = annual_sales_value / avg_inventory_value_12m if avg_inventory_value_12m > 0 else np.nan
    lead_time_coverage_pct = float((x["Stock"] >= x["Lead_Time_Demand"]).mean() * 100) if len(x) else 0
    critical_pct = float((x["Status"] == "🔴 CRITICAL").mean() * 100) if len(x) else 0
    excess_pct = float(excess_value / inventory_value * 100) if inventory_value > 0 else 0
    service_risk_pct = float(service_risk_value / inventory_value * 100) if inventory_value > 0 else 0
    purchase_to_inventory_pct = float(purchase_value / inventory_value * 100) if inventory_value > 0 else 0
    avg_lead_time = float(x["Lead_Time_Days"].mean()) if len(x) else 0
    weighted_lead_time = float(np.average(x["Lead_Time_Days"], weights=np.maximum(x["Annual_Sales"], 0))) if x["Annual_Sales"].sum() > 0 else avg_lead_time

    latest_period = "—"
    if all(c in r.columns for c in ["Year", "Month"]) and len(r):
        rp = r[["Year", "Month"]].copy()
        rp["Year"] = pd.to_numeric(rp["Year"], errors="coerce")
        rp["Month"] = pd.to_numeric(rp["Month"], errors="coerce")
        rp = rp.dropna()
        if not rp.empty:
            latest_period = f"{int(rp['Year'].max())}-{int(rp.loc[rp['Year'].eq(rp['Year'].max()), 'Month'].max()):02d}"

    kpi_rows = pd.DataFrame([
        ["Inventory Value", inventory_value, "€", "Capital currently held in inventory"],
        ["Inventory Turns", inventory_turns, "x", "Trailing-12-month sales value / average inventory value"],
        ["Aggregate Days of Cover", aggregate_cover, "days", "Current stock / aggregate average daily demand"],
        ["Lead-time Coverage", lead_time_coverage_pct, "%", "% of SKUs with stock covering lead-time demand"],
        ["Critical SKU Rate", critical_pct, "%", "% of SKUs below lead-time demand"],
        ["Excess Inventory", excess_value, "€", "Inventory above calculated required stock"],
        ["Excess Inventory Rate", excess_pct, "%", "Excess inventory value / inventory value"],
        ["Service Risk Exposure", service_risk_value, "€", "Value exposed before replenishment arrives"],
        ["Open PO Value", open_po_value, "€", "Value of currently open purchase orders"],
        ["Purchase Requirement", purchase_value, "€", "Recommended replenishment value"],
        ["Purchase / Inventory", purchase_to_inventory_pct, "%", "Purchase requirement relative to inventory"],
        ["Average Lead Time", avg_lead_time, "days", "Average supplier lead time"],
        ["Weighted Lead Time", weighted_lead_time, "days", "Lead time weighted by annual demand"],
        ["Next Month Forecast", float(x["Forecast_Next_Month"].sum()), "units", "Aggregate next-month demand forecast"],
        ["Suppliers", int(x["Supplier"].nunique()), "", "Distinct suppliers in current analysis"],
        ["Latest Period", latest_period, "", "Latest period detected in source data"],
    ], columns=["KPI", "Value", "Unit", "Definition"])

    trend = pd.DataFrame()
    if all(c in r.columns for c in ["Year", "Month", "Sales", "Stock", "Open_PO", "Unit_Cost"]):
        r["Period"] = pd.to_numeric(r["Year"], errors="coerce").fillna(0).astype(int).astype(str) + "-" + pd.to_numeric(r["Month"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
        # Value metrics must be calculated row-wise before aggregation.
        r["Sales_Value"] = r["Sales"] * r["Unit_Cost"]
        r["Inventory_Value_Row"] = r["Stock"] * r["Unit_Cost"]
        r["Open_PO_Value"] = r["Open_PO"] * r["Unit_Cost"]
        value_trend = r.groupby(["Year", "Month", "Period"], as_index=False).agg(
            Sales_Units=("Sales", "sum"),
            Sales_Value=("Sales_Value", "sum"),
            Inventory_Value=("Inventory_Value_Row", "sum"),
            Open_PO_Value=("Open_PO_Value", "sum"),
            Open_PO_Units=("Open_PO", "sum"),
            Avg_Lead_Time=("Lead_Time_Days", "mean"),
        )
        trend = value_trend.sort_values(["Year", "Month"]).tail(12).reset_index(drop=True)

    supplier = x.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU", "count"),
        Inventory_Value=("Inventory_Value", "sum"),
        Open_PO_Value=("Open_PO", lambda s: float((s * x.loc[s.index, "Unit_Cost"]).sum())),
        Purchase_Value=("Purchase_Value", "sum"),
        Service_Risk_Value=("Service_Risk_Value", "sum"),
        Excess_Inventory_Value=("Excess_Inventory_Value", "sum"),
        Critical=("Status", lambda s: int((s == "🔴 CRITICAL").sum())),
        Avg_Cover=("Days_Cover", lambda s: float(pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).mean())),
    ).sort_values(["Service_Risk_Value", "Inventory_Value"], ascending=[False, False]).reset_index(drop=True)

    abc_xyz = pd.crosstab(x["ABC"], x["XYZ"]).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    lead_bins = pd.cut(
        x["Lead_Time_Days"],
        bins=[-np.inf, 7, 14, 30, np.inf],
        labels=["≤7d", "8–14d", "15–30d", ">30d"]
    ).value_counts().reindex(["≤7d", "8–14d", "15–30d", ">30d"], fill_value=0).rename_axis("Lead_Time_Bucket").reset_index(name="SKUs")

    status = x["Status"].value_counts().rename_axis("Status").reset_index(name="SKUs")
    return {
        "kpis": kpi_rows,
        "trend": trend,
        "supplier": supplier,
        "abc_xyz": abc_xyz.reset_index().rename(columns={"ABC": "ABC_Class"}),
        "lead_bins": lead_bins,
        "status": status,
        "meta": {
            "inventory_value": inventory_value,
            "inventory_turns": inventory_turns,
            "aggregate_cover": aggregate_cover,
            "lead_time_coverage_pct": lead_time_coverage_pct,
            "critical_pct": critical_pct,
            "excess_value": excess_value,
            "excess_pct": excess_pct,
            "service_risk_value": service_risk_value,
            "open_po_value": open_po_value,
            "purchase_value": purchase_value,
            "purchase_to_inventory_pct": purchase_to_inventory_pct,
            "avg_lead_time": avg_lead_time,
            "weighted_lead_time": weighted_lead_time,
            "latest_period": latest_period,
        },
    }

def kpis(a):
    return {
        "sku": len(a),
        "inventory": float(a["Inventory_Value"].sum()),
        "purchase": float(a["Purchase_Value"].sum()),
        "critical": int((a["Status"]=="🔴 CRITICAL").sum()),
        "review": int((a["Status"]=="🟠 REVIEW").sum()),
        "excess": int((a["Status"]=="🟡 EXCESS").sum()),
        "forecast": float(a["Forecast_Next_Month"].sum()),
        "purchase_skus": int((a["Recommended_Order"]>0).sum()),
    }

def decision_text(a):
    top = a.sort_values("Decision_Score", ascending=False).head(10)
    es = st.session_state.get("language", "English") == "Spanish"
    lines = [
        tr("### Priorities for this week"),
        "",
        tr("| Priority | SKU | Action | Timing | Quantity | Risk | Confidence |"),
        "|---|---|---|---|---:|---|---|"
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        action_map = {
            "BUY_NOW": tr("Buy now"),
            "CONFIRM_PO": tr("Confirm PO"),
            "DO_NOT_BUY": tr("Do not buy"),
            "REVIEW": tr("Review"),
            "MONITOR": tr("Monitor"),
        }
        action = action_map.get(r["Action"], r["Action"])
        qty = f"{r['Recommended_Order']:.0f}" if r["Recommended_Order"] > 0 else "—"
        risk = f"{r['Days_Cover']:.1f}d {tr('cover')} / {r['Lead_Time_Days']:.0f}d LT"
        lines.append(
            f"| {i} | **{r['SKU']}** | {action} | {r['Action_Timing']} | "
            f"{qty} | {risk} | {r['Decision_Confidence']} |"
        )

    critical = a[a["Action"] == "BUY_NOW"].copy()
    excess = a[a["Action"] == "DO_NOT_BUY"].copy()
    purchase_value = float(a["Purchase_Value"].sum())
    excess_value = float(excess["Inventory_Value"].sum())

    lines += [
        "",
        f"**{tr('Total recommended purchase')}:** €{purchase_value:,.0f}",
        f"**{tr('Current excess inventory')}:** €{excess_value:,.0f}",
    ]

    if not critical.empty:
        lines.append("")
        lines.append(tr("**Immediate action:** issue/validate orders for `BUY_NOW` SKUs and confirm the delivery date with the supplier."))
    return "\n".join(lines)

def agent_purchase_tool(a):
    x = a[(a["Action"].isin(["BUY_NOW", "CONFIRM_PO"])) | (a["Recommended_Order"] > 0)].copy()
    x = x.sort_values("Decision_Score", ascending=False)
    return {
        "name": "purchase_planner",
        "purpose": "Prioritize what should be purchased or confirmed this week.",
        "kpis": {
            "recommended_purchase_value": float(x["Purchase_Value"].sum()),
            "lines": int(len(x)),
            "critical_lines": int((x["Status"] == "🔴 CRITICAL").sum()),
        },
        "rows": x[[
            "SKU","Description","Supplier","Status","Action","Action_Timing",
            "Stock","Open_PO","Days_Cover","Lead_Time_Days","Recommended_Order",
            "Purchase_Value","PO_Adequacy","Decision_Confidence"
        ]].round(2).to_dict("records"),
    }

def agent_inventory_tool(a):
    x = a[(a["Action"] == "DO_NOT_BUY") | (a["Excess_Inventory_Value"] > 0)].copy()
    x = x.sort_values("Excess_Inventory_Value", ascending=False)
    return {
        "name": "inventory_optimizer",
        "purpose": "Identify excess inventory and opportunities to stop replenishment.",
        "kpis": {
            "excess_inventory_value": float(x["Excess_Inventory_Value"].sum()),
            "sku_count": int(len(x)),
        },
        "rows": x[[
            "SKU","Description","Supplier","Stock","Required_Stock",
            "Days_Cover","Excess_Inventory_Qty","Excess_Inventory_Value",
            "Open_PO","Action"
        ]].round(2).to_dict("records"),
    }

def agent_service_tool(a):
    x = a[a["Service_Risk_Value"] > 0].copy()
    x = x.sort_values(["Service_Risk_Value","Decision_Score"], ascending=False)
    return {
        "name": "service_risk",
        "purpose": "Identify inventory gaps that can create service risk before replenishment arrives.",
        "kpis": {
            "service_risk_value": float(x["Service_Risk_Value"].sum()),
            "sku_count": int(len(x)),
        },
        "rows": x[[
            "SKU","Description","Supplier","Stock","Lead_Time_Demand",
            "Days_Cover","Lead_Time_Days","Service_Risk_Qty","Service_Risk_Value"
        ]].round(2).to_dict("records"),
    }

def agent_supplier_tool(a):
    x = a.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU","count"),
        Critical=("Status", lambda s: (s == "🔴 CRITICAL").sum()),
        Review=("Status", lambda s: (s == "🟠 REVIEW").sum()),
        Purchase_Value=("Purchase_Value","sum"),
        Inventory_Value=("Inventory_Value","sum"),
        Service_Risk_Value=("Service_Risk_Value","sum"),
        Excess_Inventory_Value=("Excess_Inventory_Value","sum"),
    )
    x["Supplier_Risk_Score"] = (
        x["Critical"] * 100 + x["Review"] * 40
        + np.log1p(x["Service_Risk_Value"]) * 5
        + np.log1p(x["Purchase_Value"]) * 2
    )
    x = x.sort_values("Supplier_Risk_Score", ascending=False)
    return {
        "name": "supplier_risk",
        "purpose": "Rank suppliers by operational risk and economic exposure.",
        "kpis": {
            "supplier_count": int(len(x)),
            "suppliers_with_critical": int((x["Critical"] > 0).sum()),
        },
        "rows": x.round(2).to_dict("records"),
    }

def agent_forecast_tool(a):
    x = a.sort_values("Forecast_Change_Pct", ascending=False)
    return {
        "name": "demand_outlook",
        "purpose": "Identify demand growth/decline that may change replenishment priorities.",
        "rising": x.head(10)[[
            "SKU","Description","Forecast_Next_Month","Forecast_Change_Pct",
            "Trend_Units_Per_Month","Demand_CV","Action"
        ]].round(2).to_dict("records"),
        "falling": x.tail(10)[[
            "SKU","Description","Forecast_Next_Month","Forecast_Change_Pct",
            "Trend_Units_Per_Month","Demand_CV","Action"
        ]].round(2).to_dict("records"),
    }

def agent_logistics_kpi_tool(a, raw=None):
    """Expose the main logistics KPIs to the Copilot without requiring the LLM."""
    inventory_value = float(a["Inventory_Value"].sum())
    sales_value = float((a["Annual_Sales"] * a["Unit_Cost"]).sum())
    avg_inventory_value = inventory_value
    if raw is not None and all(c in raw.columns for c in ["Year", "Month", "Sales", "Stock", "Unit_Cost"]):
        rr = raw.copy()
        rr["_PeriodKey"] = pd.to_numeric(rr["Year"], errors="coerce").fillna(0).astype(int) * 100 + pd.to_numeric(rr["Month"], errors="coerce").fillna(0).astype(int)
        recent_periods = sorted(rr["_PeriodKey"].unique())[-12:]
        rr = rr[rr["_PeriodKey"].isin(recent_periods)].copy()
        rr["_SalesValue"] = pd.to_numeric(rr["Sales"], errors="coerce").fillna(0) * pd.to_numeric(rr["Unit_Cost"], errors="coerce").fillna(0)
        rr["_InventoryValue"] = pd.to_numeric(rr["Stock"], errors="coerce").fillna(0) * pd.to_numeric(rr["Unit_Cost"], errors="coerce").fillna(0)
        sales_value = float(rr["_SalesValue"].sum())
        monthly_inventory = rr.groupby("_PeriodKey")["_InventoryValue"].sum()
        if not monthly_inventory.empty:
            avg_inventory_value = float(monthly_inventory.mean())
    avg_daily_demand = float(a["Avg_Daily_Demand"].sum())
    aggregate_cover = float(a["Stock"].sum() / avg_daily_demand) if avg_daily_demand > 0 else np.nan
    inventory_turns = sales_value / avg_inventory_value if avg_inventory_value > 0 else np.nan
    lead_coverage = float((a["Stock"] >= a["Lead_Time_Demand"]).mean() * 100) if len(a) else 0
    return {
        "name": "logistics_kpis",
        "purpose": "Summarize the main logistics and inventory KPIs for the current planning period.",
        "kpis": {
            "inventory_value": inventory_value,
            "inventory_turns": inventory_turns,
            "days_cover": aggregate_cover,
            "lead_time_coverage_pct": lead_coverage,
            "critical_skus": int((a["Status"] == "🔴 CRITICAL").sum()),
            "critical_pct": float((a["Status"] == "🔴 CRITICAL").mean() * 100) if len(a) else 0,
            "service_risk_value": float(a["Service_Risk_Value"].sum()),
            "excess_inventory_value": float(a["Excess_Inventory_Value"].sum()),
            "open_po_value": float((a["Open_PO"] * a["Unit_Cost"]).sum()),
            "purchase_requirement": float(a["Purchase_Value"].sum()),
            "supplier_count": int(a["Supplier"].nunique()),
            "avg_lead_time": float(a["Lead_Time_Days"].mean()) if len(a) else 0,
        },
    }

def route_agent(question, a, comparison=None, comparison_meta=None, raw=None):
    q = question.lower()
    tools = []
    if any(k in q for k in [
        "kpi", "kpis", "indicador", "indicadores", "dashboard", "logística", "logistica",
        "rotación", "rotacion", "days of cover", "cobertura", "lead time coverage"
    ]):
        tools.append(agent_logistics_kpi_tool(a, raw))
    if any(k in q for k in [
        "qué hago","que hago","qué debería hacer","que deberia hacer",
        "siguiente","next step","plan de acción","plan de accion",
        "execution","ejecutar","execute","secuencia","planning agent",
        "cómo actuar","como actuar"
    ]):
        tools.append(agent_planning_tool(a, comparison, comparison_meta))
    if any(k in q for k in ["cambio","cambió","cambio","compar","anterior","último periodo","ultimo periodo","evolución","empeor","mejoró","mejoro","vs","versus"]):
        tools.append(agent_change_monitor_tool(comparison, comparison_meta))
    if any(k in q for k in ["compr","purchase","orden","po","reponer","buy"]):
        tools.append(agent_purchase_tool(a))
    if any(k in q for k in ["exceso","sobrestock","inventory","inventario","capital"]):
        tools.append(agent_inventory_tool(a))
    if any(k in q for k in ["rotura","riesgo","servicio","stockout","nivel de servicio"]):
        tools.append(agent_service_tool(a))
    if any(k in q for k in ["proveedor","supplier","vendor"]):
        tools.append(agent_supplier_tool(a))
    if any(k in q for k in ["forecast","demanda","previsión","tendencia"]):
        tools.append(agent_forecast_tool(a))
    if not tools or any(k in q for k in ["prioridad","prioridades","resumen","esta semana","qué debería"]):
        tools = [agent_purchase_tool(a), agent_service_tool(a), agent_inventory_tool(a), agent_supplier_tool(a), agent_forecast_tool(a)]
        if any(k in q for k in ["cambio","compar","evolución","último","ultimo"]):
            tools.append(agent_change_monitor_tool(comparison, comparison_meta))
    seen = set()
    out = []
    for t in tools:
        if t["name"] not in seen:
            out.append(t)
            seen.add(t["name"])
    return out

def agent_local_response(question, a, comparison=None, comparison_meta=None, raw=None):
    tools = route_agent(question, a, comparison, comparison_meta, raw)
    lines = [tr("## Supply Chain Agent — analysis")]
    for tool in tools:
        lines.append(f"### {tool['name']}")
        if tool["name"] == "logistics_kpis":
            k = tool["kpis"]
            lines.append(
                f"{tr('Inventory')} **€{k['inventory_value']:,.0f}** · {tr('turns')} **{k['inventory_turns']:.2f}x** · "
                f"{tr('aggregate cover')} **{k['days_cover']:.1f} {tr('days')}** · {tr('lead-time coverage')} **{k['lead_time_coverage_pct']:.1f}%**."
            )
            lines.append(
                f"{tr('Service risk')} **€{k['service_risk_value']:,.0f}** · {tr('excess')} **€{k['excess_inventory_value']:,.0f}** · "
                f"{tr('open POs')} **€{k['open_po_value']:,.0f}** · {tr('recommended purchase')} **€{k['purchase_requirement']:,.0f}**."
            )
        elif tool["name"] == "purchase_planner":
            k = tool["kpis"]
            lines.append(f"{tr('Recommended purchase')}: **€{k['recommended_purchase_value']:,.0f}** · {k['lines']} {tr('lines')} · {k['critical_lines']} {tr('critical') }.")
            lines += [f"- **{r['SKU']}** → {r['Action']} · {r['Recommended_Order']:.0f} {tr('units')} · {tr('cover')} {r['Days_Cover']:.1f}d · LT {r['Lead_Time_Days']:.0f}d." for r in tool["rows"][:8]]
        elif tool["name"] == "inventory_optimizer":
            k = tool["kpis"]
            lines.append(f"{tr('Estimated excess')}: **€{k['excess_inventory_value']:,.0f}** {tr('across')} {k['sku_count']} SKUs.")
        elif tool["name"] == "service_risk":
            k = tool["kpis"]
            lines.append(f"{tr('Service exposure')}: **€{k['service_risk_value']:,.0f}** {tr('across')} {k['sku_count']} SKUs.")
        elif tool["name"] == "supplier_risk":
            lines += [f"- **{r['Supplier']}** → {tr('critical')} {int(r['Critical'])}, {tr('purchase')} €{r['Purchase_Value']:,.0f}, {tr('service risk lower')} €{r['Service_Risk_Value']:,.0f}." for r in tool["rows"][:8]]
        elif tool["name"] == "demand_outlook":
            lines.append(tr("Largest demand increases:"))
            lines += [f"- **{r['SKU']}** → {tr('forecast')} {r['Forecast_Next_Month']:.0f} {tr('units')} ({r['Forecast_Change_Pct']:+.1f}%)." for r in tool["rising"][:5]]
        elif tool["name"] == "planning_agent":
            k = tool["kpis"]
            lines.append(f"{tr('Execution plan')}: **{k['immediate_count']} {tr('immediate actions')}**, €{k['purchase_value']:,.0f} {tr('purchase exposure')} and €{k['service_risk_addressed']:,.0f} {tr('service risk potentially addressable')}.")
            lines += [f"- **{tr('Step')} {r['Execution_Priority']} — {r['SKU']}**: {r['Execution_Task']} · {r['Dependency']} · {r['Action_Timing']}." for r in tool["rows"][:8]]
        elif tool["name"] == "change_monitor":
            if not tool["kpis"].get("available"):
                lines.append(tr("No comparable periods are available."))
            else:
                k = tool["kpis"]
                lines.append(f"{tr('Comparison')} **{k['previous_period']} → {k['current_period']}**: {tr('purchase')} {k['purchase_delta']:+,.0f} €, {tr('service risk')} {k['service_risk_delta']:+,.0f} €, {tr('action changes')} {k['action_changes']}.")
                lines += [f"- **{r['SKU']}** → {r['Change_Classification']} · {tr('coverage')} {r['Days_Cover_Delta']:+.1f}d · {tr('purchase')} {r['Purchase_Value_Delta']:+,.0f} € · {r['Change_Reason']}" for r in tool["rows"][:8]]
    return "\n".join(lines)

def build_context(a):
    cols = [
        "SKU","Description","Supplier","Status","Stock","Open_PO",
        "Days_Cover","Lead_Time_Days","Lead_Time_Gap_Days","Action","Action_Timing","Decision_Confidence",
        "Recommended_Order","Purchase_Value",
        "Forecast_Next_Month","Forecast_Change_Pct" if "Forecast_Change_Pct" in a else "Forecast_Next_Month",
        "ABC_XYZ","Inventory_Value","Demand_CV"
    ]
    cols = list(dict.fromkeys([c for c in cols if c in a.columns]))
    return a[cols].round(2).to_csv(index=False)

def _make_openai_client(api_key):
    """Create an OpenAI client using Streamlit Secrets or environment settings."""
    kwargs = {"api_key": api_key.strip()}

    project_id = os.getenv("OPENAI_PROJECT_ID", "").strip()
    org_id = os.getenv("OPENAI_ORG_ID", "").strip()
    try:
        project_id = project_id or str(st.secrets.get("OPENAI_PROJECT_ID", "")).strip()
        org_id = org_id or str(st.secrets.get("OPENAI_ORG_ID", "")).strip()
    except Exception:
        pass

    if project_id:
        kwargs["project"] = project_id
    if org_id:
        kwargs["organization"] = org_id

    return OpenAI(**kwargs)

def _openai_error_info(exc):
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    code = None
    error_type = None

    if isinstance(body, dict):
        payload = body.get("error", body)
        if isinstance(payload, dict):
            code = payload.get("code")
            error_type = payload.get("type")

    return status, code, error_type

def test_openai_connection(api_key, model):
    if not api_key:
        return False, ("No OPENAI_API_KEY configured." if st.session_state.get("language", "English") == "English" else "No hay API key configurada."), {}
    if OpenAI is None:
        return False, ("The OpenAI library is not installed." if st.session_state.get("language", "English") == "English" else "La librería OpenAI no está instalada."), {}

    try:
        client = _make_openai_client(api_key)

        # Authentication/permissions test independent from the selected model.
        client.models.list()

        # Small Responses API smoke test.
        response = client.responses.create(
            model=model,
            input=("Respond only: connection OK" if st.session_state.get("language", "English") == "English" else "Responde únicamente: conexión OK")
        )
        return True, response.output_text, {}

    except Exception as exc:
        status, code, error_type = _openai_error_info(exc)
        return False, str(exc), {
            "http_status": status,
            "error_code": code,
            "error_type": error_type,
            "exception": type(exc).__name__,
        }

def ai_chat(question, a, api_key=None, model="gpt-5.6-luna", comparison=None, comparison_meta=None, raw=None):
    tool_results = route_agent(question, a, comparison, comparison_meta, raw)

    if not api_key or OpenAI is None:
        return agent_local_response(question, a, comparison, comparison_meta, raw)

    try:
        client = _make_openai_client(api_key)
        context = build_context(a)
        response = client.responses.create(
            model=model,
            instructions=(
                (
                    "You are a senior Supply Chain agent. Respond in English and operationally. "
                    "Use the structured tool results. Do not invent numbers. Do not execute purchases. "
                    "For purchasing distinguish BUY_NOW from CONFIRM_PO. If an open PO exists, validate its adequacy before recommending a new purchase. "
                    "For excess quantify potentially releasable inventory value. For service quantify exposed units and value. "
                    "For suppliers explain risk concentration. For forecast highlight changes that may alter decisions. "
                    "For logistics KPIs use the logistics_kpis tool and clearly distinguish lead-time coverage from OTIF/fill rate. "
                    "For comparisons explain what improved, worsened or changed action between periods and quantify deltas. "
                    "For executive questions use: Summary → Top 3 priorities → Actions → Risks/Assumptions."
                ) if st.session_state.get("language", "English") == "English" else (
                    "Eres un agente senior de Supply Chain. Responde en español y de forma operativa. "
                    "Usa los resultados estructurados de las herramientas. No inventes números. No ejecutes compras. "
                    "Para compras distingue BUY_NOW de CONFIRM_PO. Si existe una PO abierta, confirma su adecuación antes de recomendar una nueva compra. "
                    "Para exceso cuantifica el valor de inventario potencialmente liberable. Para servicio cuantifica unidades y valor expuesto. "
                    "Para proveedores explica la concentración de riesgo. Para forecast señala cambios que puedan modificar decisiones. "
                    "Para KPIs logísticos usa la herramienta logistics_kpis y distingue claramente cobertura de lead time de OTIF/fill rate. "
                    "Para comparación explica qué ha mejorado, empeorado o cambiado de acción entre periodos y cuantifica los deltas. "
                    "En preguntas ejecutivas: Resumen → Top 3 prioridades → Acciones → Riesgos/Supuestos."
                )
            ),
            input=f"PREGUNTA:\\n{question}\\n\\nHERRAMIENTAS:\\n{tool_results}\\n\\nDATASET:\\n{context}"
        )
        return response.output_text
    except AuthenticationError as exc:
        status, code, error_type = _openai_error_info(exc)
        return f"### ⚠️ Autenticación OpenAI fallida\\nHTTP `{status or 'desconocido'}` · code `{code or 'desconocido'}` · type `{error_type or 'desconocido'}`"
    except RateLimitError as exc:
        status, code, error_type = _openai_error_info(exc)
        return f"### ⚠️ Cuota/límite OpenAI\\nHTTP `{status or 'desconocido'}` · code `{code or 'desconocido'}` · type `{error_type or 'desconocido'}`"
    except APIError as exc:
        status, code, error_type = _openai_error_info(exc)
        return f"### ⚠️ Error OpenAI\\nHTTP `{status or 'desconocido'}` · code `{code or 'desconocido'}` · type `{error_type or 'desconocido'}`"
    except Exception as exc:
        return f"### ⚠️ Error del agente\\n`{type(exc).__name__}: {str(exc)}`"


def export_purchase(a):
    x = a[a["Recommended_Order"] > 0].copy()
    x["Estimated_PO_Value"] = x["Purchase_Value"]
    cols = ["Supplier","SKU","Description","Recommended_Order","Unit_Cost","Estimated_PO_Value","Lead_Time_Days"]
    return x[cols].sort_values(["Supplier","Estimated_PO_Value"], ascending=[True,False])

# -----------------------------
# Manufacturing / MRP engine
# -----------------------------
def normalize_bom(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Parent_SKU","Component_SKU","Qty_Per","Scrap_Pct"])
    aliases = {
        "parent_sku":"Parent_SKU", "parent":"Parent_SKU", "finished_good":"Parent_SKU", "finished_good_sku":"Parent_SKU",
        "component_sku":"Component_SKU", "component":"Component_SKU", "child_sku":"Component_SKU", "material":"Component_SKU",
        "qty_per":"Qty_Per", "quantity_per":"Qty_Per", "qty":"Qty_Per", "quantity":"Qty_Per",
        "scrap_pct":"Scrap_Pct", "scrap":"Scrap_Pct", "waste_pct":"Scrap_Pct",
    }
    rename={}
    for c in df.columns:
        key=str(c).strip().lower().replace(" ","_")
        if key in aliases: rename[c]=aliases[key]
    x=df.rename(columns=rename).copy()
    if not all(c in x.columns for c in ["Parent_SKU","Component_SKU","Qty_Per"]):
        return pd.DataFrame()
    x["Parent_SKU"]=x["Parent_SKU"].astype(str).str.strip()
    x["Component_SKU"]=x["Component_SKU"].astype(str).str.strip()
    x["Qty_Per"]=pd.to_numeric(x["Qty_Per"],errors="coerce")
    if "Scrap_Pct" not in x.columns: x["Scrap_Pct"]=0.0
    x["Scrap_Pct"]=pd.to_numeric(x["Scrap_Pct"],errors="coerce").fillna(0).clip(lower=0,upper=100)
    x=x[(x["Parent_SKU"]!="")&(x["Component_SKU"]!="")&(x["Qty_Per"]>0)].copy()
    x["Effective_Qty_Per"]=x["Qty_Per"]*(1+x["Scrap_Pct"]/100)
    return x[["Parent_SKU","Component_SKU","Qty_Per","Scrap_Pct","Effective_Qty_Per"]]

def _mrp_template_bytes():
    df=pd.DataFrame([
        ["FG-001","RM-001",2.0,2.0],
        ["FG-001","RM-002",1.0,0.0],
        ["FG-002","RM-001",1.5,1.0],
    ],columns=["Parent_SKU","Component_SKU","Qty_Per","Scrap_Pct"])
    if xlsxwriter is None:
        return df.to_csv(index=False).encode("utf-8"), "csv"
    buf=io.BytesIO(); wb=xlsxwriter.Workbook(buf,{"in_memory":True}); ws=wb.add_worksheet("BOM Template" if st.session_state.get("language","English")=="English" else "Plantilla BOM")
    head=wb.add_format({"bold":True,"bg_color":"#17365D","font_color":"#FFFFFF"})
    for j,c in enumerate(df.columns): ws.write(0,j,c,head)
    for i,row in enumerate(df.itertuples(index=False),1):
        for j,v in enumerate(row): ws.write(i,j,v)
    ws.set_column("A:B",20); ws.set_column("C:D",14); ws.freeze_panes(1,0); wb.close()
    return buf.getvalue(), "xlsx"

def _mrp_round_qty(qty, lot_size):
    qty=float(max(qty,0)); lot=float(lot_size or 0)
    if qty<=0: return 0.0
    return math.ceil(qty/lot)*lot if lot>0 else qty

def run_mrp(raw, full_analysis, bom, parent_skus=None, horizon=6, demand_growth=0.0, production_safety_days=5, use_open_po=True):
    if bom is None or bom.empty: return pd.DataFrame(), pd.DataFrame(), {"parents":0,"components":0,"shortages":0,"purchase":0.0}
    b=bom.copy()
    master=full_analysis.copy()
    master["SKU"]=master["SKU"].astype(str)
    if parent_skus:
        parents=[str(x) for x in parent_skus]
    else:
        parents=sorted(b["Parent_SKU"].unique().tolist())
    parents=[p for p in parents if p in set(b["Parent_SKU"])]
    if not parents: return pd.DataFrame(), pd.DataFrame(), {"parents":0,"components":0,"shortages":0,"purchase":0.0}
    periods=[]
    available_periods = _periods_from_raw(raw)
    if available_periods:
        y, m = _period_tuple(available_periods[-1])
        base = pd.Period(f"{y}-{m:02d}", freq="M") + 1
    else:
        base = pd.Timestamp.today().to_period("M")
    for i in range(horizon): periods.append(str(base+i))
    # Parent MPS based on next-month forecast and a controlled growth assumption.
    parent_master=master.set_index("SKU")
    mps=[]
    for p in parents:
        if p not in parent_master.index: continue
        r=parent_master.loc[p]
        base_fc=float(r.get("Forecast_Next_Month",0) or 0)
        avg_daily=float(r.get("Avg_Daily_Demand",0) or 0)
        safety=avg_daily*production_safety_days
        stock=float(r.get("Stock",0) or 0)
        open_po=float(r.get("Open_PO",0) or 0) if use_open_po else 0
        for i,period in enumerate(periods):
            demand=max(0,base_fc*((1+demand_growth)**i))
            receipt=open_po if i==0 else 0
            planned=max(0,demand+safety-stock-receipt) if i==0 else max(0,demand+safety)
            # For later months, carry the prior projected balance implicitly by using demand + safety; MRP components remain conservative.
            if i>0: planned=demand
            mps.append({"Parent_SKU":p,"Period":period,"Gross_Requirement":demand,"Planned_Production":planned})
    mps_df=pd.DataFrame(mps)
    if mps_df.empty: return pd.DataFrame(), pd.DataFrame(), {"parents":0,"components":0,"shortages":0,"purchase":0.0}
    # Explode MPS through BOM.
    merged=mps_df.merge(b[["Parent_SKU","Component_SKU","Effective_Qty_Per"]],on="Parent_SKU",how="inner")
    merged["Gross_Requirement"]=merged["Planned_Production"]*merged["Effective_Qty_Per"]
    gross=merged.groupby(["Component_SKU","Period"],as_index=False)["Gross_Requirement"].sum()
    comp_master=master.set_index("SKU")
    rows=[]
    for comp in sorted(gross["Component_SKU"].unique()):
        if comp in comp_master.index:
            r=comp_master.loc[comp]
            opening=float(r.get("Stock",0) or 0)
            open_po=float(r.get("Open_PO",0) or 0) if use_open_po else 0
            lead=float(r.get("Lead_Time_Days",0) or 0)
            moq=float(r.get("MOQ",0) or 0)
            unit_cost=float(r.get("Unit_Cost",0) or 0)
            supplier=str(r.get("Supplier",""))
        else:
            opening=open_po=lead=moq=unit_cost=0.0; supplier="Unknown"
        prev=opening
        for i,period in enumerate(periods):
            gr=float(gross[(gross.Component_SKU==comp)&(gross.Period==period)]["Gross_Requirement"].sum())
            receipt=open_po if (i==0 and use_open_po) else 0.0
            projected_before=prev+receipt-gr
            net=max(0.0,-projected_before)
            por=_mrp_round_qty(net,moq)
            projected_end=projected_before+por
            shortage=max(0.0,-projected_before) if por==0 else 0.0
            rows.append({"Component_SKU":comp,"Period":period,"Opening_Inventory":prev,"Gross_Requirement":gr,"Scheduled_Receipts":receipt,"Net_Requirement":net,"Planned_Order_Receipt":por,"Projected_Ending_Inventory":projected_end,"Shortage":shortage,"Supplier":supplier,"Lead_Time_Days":lead,"MOQ":moq,"Unit_Cost":unit_cost,"Planned_Purchase_Value":por*unit_cost})
            prev=projected_end
    mrp=pd.DataFrame(rows)
    # Release period is receipt period shifted by lead time in whole months.
    if not mrp.empty:
        mrp["Release_Period"]=mrp.apply(lambda r: str(pd.Period(r["Period"],freq="M")-max(0,math.ceil(float(r["Lead_Time_Days"])/30))),axis=1)
        mrp["Action"]=np.where(mrp["Planned_Order_Receipt"]>0,"BUY",np.where(mrp["Shortage"]>0,"SHORTAGE","NONE"))
        mrp["MRP_Type"]="PURCHASE"
    # Parent-level MPS summary.
    mps_summary=mps_df.groupby("Period",as_index=False).agg(Parents=("Parent_SKU","nunique"),Gross_Demand=("Gross_Requirement","sum"),Planned_Production=("Planned_Production","sum"))
    mps_summary["MRP_Type"]="MAKE"
    meta={"parents":len(parents),"components":int(mrp["Component_SKU"].nunique()) if not mrp.empty else 0,"shortages":int((mrp["Shortage"]>0).sum()) if not mrp.empty else 0,"purchase":float(mrp["Planned_Purchase_Value"].sum()) if not mrp.empty else 0.0}
    return mps_df,mrp,meta

def _mrp_export_bytes(mps, mrp, meta, language="English"):
    if xlsxwriter is None: return None
    buf=io.BytesIO(); wb=xlsxwriter.Workbook(buf,{"in_memory":True})
    title_fmt=wb.add_format({"bold":True,"font_size":18,"font_color":"#FFFFFF","bg_color":"#17365D"})
    head=wb.add_format({"bold":True,"bg_color":"#D9EAF7","border":1})
    money=wb.add_format({"num_format":"€#,##0.00"}); qty=wb.add_format({"num_format":"#,##0.00"})
    ws=wb.add_worksheet("MRP Summary" if language=="English" else "Resumen MRP")
    ws.merge_range("A1:F1","Manufacturing MRP" if language=="English" else "MRP de fabricación",title_fmt)
    labels=["Parent SKUs","Components","Shortages","Planned purchase"] if language=="English" else ["SKUs padre","Componentes","Faltantes","Compra planificada"]
    vals=[meta["parents"],meta["components"],meta["shortages"],meta["purchase"]]
    for j,(l,v) in enumerate(zip(labels,vals)): ws.write(2,j,l,head); ws.write(3,j,v,money if j==3 else None)
    if not mps.empty:
        w2=wb.add_worksheet("MPS" if language=="English" else "MPS Producción")
        cols=["Parent_SKU","Period","Gross_Requirement","Planned_Production"]
        if language=="Spanish": cols=["SKU padre","Periodo","Necesidad bruta","Producción planificada"]
        out=mps[["Parent_SKU","Period","Gross_Requirement","Planned_Production"]].copy(); out.columns=cols
        for j,c in enumerate(out.columns): w2.write(0,j,c,head)
        for i,row in enumerate(out.itertuples(index=False),1):
            for j,v in enumerate(row): w2.write(i,j,v,money if False else qty if j>=2 else None)
        w2.set_column(0,1,18); w2.set_column(2,3,20); w2.freeze_panes(1,0)
    if not mrp.empty:
        w3=wb.add_worksheet("MRP Detail" if language=="English" else "Detalle MRP")
        out=mrp[["Component_SKU","Period","Opening_Inventory","Gross_Requirement","Scheduled_Receipts","Net_Requirement","Planned_Order_Receipt","Projected_Ending_Inventory","Shortage","Supplier","Lead_Time_Days","MOQ","Unit_Cost","Planned_Purchase_Value","Release_Period","Action"]].copy()
        if language=="Spanish": out.columns=["SKU componente","Periodo","Inventario inicial","Necesidad bruta","Recepciones programadas","Necesidad neta","Recepción planificada","Inventario final proyectado","Faltante","Proveedor","Lead time (días)","MOQ","Coste unitario","Valor compra planificada","Periodo lanzamiento","Acción"]
        for j,c in enumerate(out.columns): w3.write(0,j,c,head)
        for i,row in enumerate(out.itertuples(index=False),1):
            for j,v in enumerate(row):
                fmt=money if j in [12,13] else qty if j in [2,3,4,5,6,7,8,10,11] else None
                w3.write(i,j,v,fmt)
        w3.set_column(0,1,18); w3.set_column(2,8,20); w3.set_column(9,9,20); w3.set_column(10,15,18); w3.freeze_panes(1,0)
    wb.close(); return buf.getvalue()

# -----------------------------
# Session state
# -----------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "copilot_prefill" not in st.session_state:
    st.session_state.copilot_prefill = ""
if "scenario_library" not in st.session_state:
    st.session_state.scenario_library = {}
if "mrp_bom" not in st.session_state:
    st.session_state.mrp_bom = None


# -----------------------------
# Internationalization (EN / ES)
# -----------------------------
if "language" not in st.session_state:
    st.session_state.language = "English"

_TRANSLATIONS = {
    "Spanish": {
        "Decision Intelligence for planners · V2.0.13": "Inteligencia de decisiones para planners · V2.0.13",
        "Historical demand and inventory": "Histórico de demanda e inventario",
        "Upload historical demand and inventory data for analysis. CSV and Excel are supported.": "Carga datos históricos de demanda e inventario para ejecutar el análisis. Se admiten CSV y Excel.",
        "Safety stock floor (days)": "Stock de seguridad mínimo (días)",
        "Service level": "Nivel de servicio",
        "Model": "Modelo",
        "Current models available in the Responses API.": "Modelos actuales disponibles en la Responses API.",
        "Test another key for this session only": "Probar otra clave solo durante esta sesión",
        "Diagnose OpenAI connection": "Diagnosticar conexión con OpenAI",
        "Load demo": "Cargar demo",
        "Current dataset": "Dataset actual",
        "Period": "Periodo",
        "No critical SKUs under the current planning parameters.": "No hay SKUs críticos con los parámetros actuales de planificación.",
        "Safety stock floor": "Stock de seguridad mínimo",
        "Service level target": "Objetivo de nivel de servicio",
        "These parameters affect safety stock, coverage and purchase recommendations.": "Estos parámetros afectan al stock de seguridad, la cobertura y las recomendaciones de compra.",
        "Current period": "Periodo actual",
        "Compare with": "Comparar con",
        "What should the planner do now?": "¿Qué debería hacer ahora el planner?",
        "Why these actions?": "¿Por qué estas acciones?",
        "Purchase plan by supplier": "Plan de compras por proveedor",
        "No purchase orders recommended.": "No se recomiendan órdenes de compra.",
        "Continue with Copilot": "Continuar con Copilot",
        "Prepared question for Copilot. Go to the 🤖 Copilot tab to run it.": "Pregunta preparada para Copilot. Ve a la pestaña 🤖 Copilot para ejecutarla.",
        "Critical inventory exposure": "Exposición de inventario crítico",
        "Excess inventory": "Exceso de inventario",
        "Immediate actions": "Acciones inmediatas",
        "Purchase need": "Necesidad de compra",
        "Inventory": "Inventario",
        "Next month": "Próximo mes",
        "Logistics KPI Dashboard": "Dashboard de KPIs logísticos",
        "Executive view of inventory, service exposure, replenishment, supplier exposure and logistics efficiency.": "Vista ejecutiva de inventario, exposición de servicio, reposición, exposición por proveedor y eficiencia logística.",
        "Sales value vs inventory value": "Valor de ventas vs valor de inventario",
        "No monthly history available for the trend chart.": "No hay histórico mensual disponible para el gráfico de tendencia.",
        "Inventory health": "Salud del inventario",
        "Supplier exposure": "Exposición por proveedor",
        "Lead-time profile": "Perfil de lead time",
        "ABC / XYZ portfolio": "Cartera ABC / XYZ",
        "Logistics KPI catalogue": "Catálogo de KPIs logísticos",
        "What the dashboard is telling the planner": "Qué está indicando el dashboard al planner",
        "Investigate with Copilot": "Investigar con Copilot",
        "Service and coverage KPIs are planning proxies derived from inventory, demand and lead-time data; they are not OTIF or customer fill-rate measurements unless those source fields are provided.": "Los KPIs de servicio y cobertura son indicadores de planificación derivados de inventario, demanda y lead time; no son mediciones OTIF ni fill rate de cliente salvo que esos campos estén disponibles en los datos fuente.",
        "Planning Agent": "Agente de planificación",
        "Turns the decision engine into an ordered sequence of planner actions.": "Convierte el motor de decisiones en una secuencia ordenada de acciones para el planner.",
        "Recommended execution sequence": "Secuencia de ejecución recomendada",
        "Planner rationale": "Justificación para el planner",
        "Export Planning Agent": "Exportar Planning Agent",
        "Demand outlook": "Perspectiva de demanda",
        "Segmentation": "Segmentación",
        "Policy simulator": "Simulador de políticas",
        "Demand & supply assumptions": "Supuestos de demanda y suministro",
        "Demand multiplier": "Multiplicador de demanda",
        "Demand volatility multiplier": "Multiplicador de volatilidad de demanda",
        "Available stock multiplier": "Multiplicador de stock disponible",
        "Open PO reliability": "Fiabilidad de las órdenes abiertas",
        "Unit cost multiplier": "Multiplicador del coste unitario",
        "MOQ multiplier": "Multiplicador de MOQ",
        "Planning policy": "Política de planificación",
        "Lead time buffer (days)": "Buffer de lead time (días)",
        "Scenario impact": "Impacto del escenario",
        "Scenario decision impact": "Impacto del escenario en las decisiones",
        "Current policy": "Política actual",
        "Scenario": "Escenario",
        "Change": "Cambio",
        "Metric": "Métrica",
        "Action changes": "Cambios de acción",
        "Service risk": "Riesgo de servicio",
        "Excess inventory": "Exceso de inventario",
        "Median days cover": "Cobertura mediana (días)",
        "Scenario results are simulations only. Validate the impact before changing the live planning policy.": "Los resultados del escenario son únicamente simulaciones. Valida el impacto antes de cambiar la política de planificación activa.",
        "Simulate decisions before changing the policy.": "Simula decisiones antes de cambiar la política.",
        "Scenario comparison": "Comparación de escenarios",
        "Save scenario": "Guardar escenario",
        "Scenario slot": "Espacio de escenario",
        "Scenario name": "Nombre del escenario",
        "Scenario A": "Escenario A",
        "Scenario B": "Escenario B",
        "Scenario C": "Escenario C",
        "Saved scenarios": "Escenarios guardados",
        "No saved scenarios yet.": "Todavía no hay escenarios guardados.",
        "Current simulation": "Simulación actual",
        "Save the current settings to compare them with other scenarios.": "Guarda la configuración actual para compararla con otros escenarios.",
        "Compare scenarios": "Comparar escenarios",
        "Scenario comparison requires at least one saved scenario.": "La comparación de escenarios requiere al menos un escenario guardado.",
        "Selected scenarios": "Escenarios seleccionados",
        "Load": "Cargar",
        "Delete": "Eliminar",
        "Scenario KPI comparison": "Comparación de KPIs de escenarios",
        "Purchase need": "Necesidad de compra",
        "Critical SKUs": "SKUs críticos",
        "Required stock value": "Valor de stock requerido",
        "Service risk": "Riesgo de servicio",
        "Excess inventory": "Exceso de inventario",
        "Median days cover": "Cobertura mediana (días)",
        "Action changes": "Cambios de acción",
        "Scenario parameters": "Parámetros del escenario",
        "Saved": "Guardado",
        "Scenario saved successfully.": "Escenario guardado correctamente.",
        "Scenario deleted.": "Escenario eliminado.",
        "Save": "Guardar",
        "Compare": "Comparar",
        "Reset comparison": "Restablecer comparación",
        "Scenario": "Escenario",
        "Lead time multiplier": "Multiplicador de lead time",
        "Data Quality": "Calidad de datos",
        "Checks the data before operational decisions are used.": "Comprueba los datos antes de utilizar decisiones operativas.",
        "Critical data-quality issues detected.": "Se han detectado problemas críticos de calidad de datos.",
        "Data-quality warnings detected. Review them before issuing purchase decisions.": "Se han detectado avisos de calidad de datos. Revísalos antes de emitir decisiones de compra.",
        "All current data-quality checks passed.": "Todas las comprobaciones actuales de calidad de datos han sido superadas.",
        "Latest period": "Último periodo",
        "Checks completed": "Comprobaciones realizadas",
        "Export Data Quality": "Exportar calidad de datos",
        "HTML provides the management-ready visual report; Excel provides editable quality checks and KPI summary.": "HTML proporciona el informe visual preparado para dirección; Excel proporciona comprobaciones editables y resumen de KPIs.",
        "Weekly Action Plan": "Plan de acción semanal",
        "Planner-ready worklist generated by the Decision Engine.": "Lista de trabajo preparada para el planner y generada por el motor de decisiones.",
        "Supplier follow-up": "Seguimiento de proveedores",
        "Export Action Plan": "Exportar plan de acción",
        "What changed?": "¿Qué ha cambiado?",
        "Need at least two historical periods to compare evolution.": "Se necesitan al menos dos periodos históricos para comparar la evolución.",
        "API key detected ·": "API key detectada ·",
        "No OPENAI_API_KEY configured. Local mode will be used.": "No hay OPENAI_API_KEY configurada. Se utilizará el modo local.",
        "Manual API key": "API key manual",
        "critical SKUs require attention.": "SKUs críticos requieren atención.",
        "Planning assumptions": "Parámetros de planificación",
    "Global filters": "Filtros globales",
    "Filter the operational views without changing the source dataset.": "Filtra las vistas operativas sin modificar el dataset fuente.",
    "SKU / description search": "Buscar SKU / descripción",
    "Supplier": "Proveedor",
    "Inventory status": "Estado del inventario",
    "ABC class": "Clase ABC",
    "XYZ class": "Clase XYZ",
    "All statuses": "Todos los estados",
    "All suppliers": "Todos los proveedores",
    "All ABC classes": "Todas las clases ABC",
    "All XYZ classes": "Todas las clases XYZ",
    "Reset filters": "Restablecer filtros",
    "Active filters": "Filtros activos",
    "Showing": "Mostrando",
    "of": "de",
    "filtered SKUs": "SKUs filtrados",
    "Clear search": "Borrar búsqueda",
    "No SKUs match the selected filters.": "Ningún SKU coincide con los filtros seleccionados.",
    "None": "Ninguno",
    "Search": "Búsqueda",
    }
}

_TRANSLATIONS["Spanish"].update({
    "MRP": "MRP",
    "Material Requirements Planning": "Planificación de necesidades de materiales",
    "Convert demand into a manufacturing and component plan.": "Convierte la demanda en un plan de fabricación y necesidades de componentes.",
    "BOM master data": "Datos maestros de BOM",
    "Upload BOM": "Cargar BOM",
    "BOM template": "Plantilla BOM",
    "Download BOM template": "Descargar plantilla BOM",
    "BOM uploaded successfully.": "BOM cargada correctamente.",
    "BOM file must contain Parent_SKU, Component_SKU and Qty_Per columns.": "El archivo BOM debe contener las columnas Parent_SKU, Component_SKU y Qty_Per.",
    "No BOM loaded. Use the template to prepare your manufacturing structure.": "No hay ninguna BOM cargada. Utiliza la plantilla para preparar la estructura de fabricación.",
    "MRP planning parameters": "Parámetros de planificación MRP",
    "Planning horizon (months)": "Horizonte de planificación (meses)",
    "Monthly demand growth": "Crecimiento mensual de la demanda",
    "Production safety stock (days)": "Stock de seguridad de producción (días)",
    "Use open POs as month-1 receipts": "Usar POs abiertas como recepciones del mes 1",
    "Parent SKUs": "SKUs padre",
    "All parent SKUs": "Todos los SKUs padre",
    "Run MRP": "Ejecutar MRP",
    "MRP results": "Resultados MRP",
    "Planned production": "Producción planificada",
    "Component requirements": "Necesidades de componentes",
    "Net requirements": "Necesidades netas",
    "Planned order receipts": "Recepciones planificadas",
    "Planned order releases": "Lanzamientos planificados",
    "Projected ending inventory": "Inventario proyectado final",
    "Gross requirements": "Necesidades brutas",
    "Scheduled receipts": "Recepciones programadas",
    "Opening inventory": "Inventario inicial",
    "Shortage": "Faltante",
    "MRP exception messages": "Mensajes de excepción MRP",
    "Material shortages requiring action.": "Faltantes de material que requieren acción.",
    "No material shortages detected in the simulated horizon.": "No se han detectado faltantes de material en el horizonte simulado.",
    "Manufacturing KPIs": "KPIs de fabricación",
    "Manufacturing orders": "Órdenes de fabricación",
    "Components with shortage": "Componentes con faltante",
    "Total component requirement": "Necesidad total de componentes",
    "Total planned purchase": "Compra total planificada",
    "MRP coverage": "Cobertura MRP",
    "BOM lines": "Líneas BOM",
    "Parent SKU": "SKU padre",
    "Component SKU": "SKU componente",
    "Qty per": "Cantidad por unidad",
    "Scrap %": "% merma",
    "Component": "Componente",
    "Period": "Periodo",
    "MRP Type": "Tipo MRP",
    "Buy": "Comprar",
    "Make": "Fabricar",
    "Action": "Acción",
    "Reason": "Motivo",
    "Supplier": "Proveedor",
    "Lead time": "Lead time",
    "MOQ": "MOQ",
    "Unit cost": "Coste unitario",
    "Export MRP": "Exportar MRP",
    "MRP workbook": "Libro MRP",
    "The MRP engine uses the selected finished-goods forecast, BOM quantities, current component inventory and open POs to calculate time-phased material needs.": "El motor MRP utiliza el forecast seleccionado de producto terminado, las cantidades de BOM, el inventario actual de componentes y las POs abiertas para calcular las necesidades de material por periodo.",
    "MRP is a planning simulation. Validate BOMs, routings, calendars and supplier due dates before execution.": "MRP es una simulación de planificación. Valida las BOM, rutas, calendarios y fechas de entrega de proveedores antes de ejecutar órdenes.",
    "No parent SKUs are available in the uploaded BOM.": "No hay SKUs padre disponibles en la BOM cargada.",
    "BOM components not found in the dataset": "Componentes de la BOM no encontrados en el dataset",
    "BOM parents not found in the dataset": "SKUs padre de la BOM no encontrados en el dataset",
    "Clear previous MRP results after loading a new BOM.": "Borra los resultados MRP anteriores al cargar una nueva BOM.",
    "Invalid BOM rows were removed.": "Se han eliminado filas BOM no válidas.",
    "Download": "Descargar",
    "Loaded BOM": "BOM cargada",
    "Max scrap %": "% merma máxima",
    "BOM upload failed": "Error al cargar la BOM",
    "Release Period": "Periodo de lanzamiento",
    "BOM fields: Parent_SKU, Component_SKU, Qty_Per, Scrap_Pct": "Campos BOM: Parent_SKU, Component_SKU, Qty_Per, Scrap_Pct",
    "Components": "Componentes",
    "rows": "filas",
})

def tr(text):
    if text is None:
        return text
    if st.session_state.get("language", "English") == "Spanish":
        return _TRANSLATIONS["Spanish"].get(str(text), str(text))
    return str(text)

_STATUS_LABELS_ES = {
    "🔴 CRITICAL": "🔴 CRÍTICO",
    "🟠 REVIEW": "🟠 REVISAR",
    "🟡 EXCESS": "🟡 EXCESO",
    "🟢 OK": "🟢 OK",
}

def status_display_options():
    canonical = ["🔴 CRITICAL", "🟠 REVIEW", "🟡 EXCESS", "🟢 OK"]
    if st.session_state.get("language", "English") == "Spanish":
        return [_STATUS_LABELS_ES[x] for x in canonical]
    return canonical

def status_from_display(value):
    if st.session_state.get("language", "English") == "Spanish":
        reverse = {v:k for k,v in _STATUS_LABELS_ES.items()}
        return reverse.get(value, value)
    return value

# Complete UI vocabulary for the bilingual interface. Calculations remain language-neutral.
_TRANSLATIONS["Spanish"].update({
    "The API key is never displayed in full or stored in GitHub.": "La API key nunca se muestra completa ni se guarda en GitHub.",
    "❌ Unable to validate the connection.": "❌ No se pudo validar la conexión.",
    "✅ OpenAI connected": "✅ OpenAI conectado",
    "⚙️ Planning assumptions": "⚙️ Parámetros de planificación",
    "🔄 Period comparison": "🔄 Comparación de periodos",
    "days": "días",
    "Current": "Actual", "Previous": "Anterior",
    "From raw supply-chain data to prioritized decisions · V2.0.13": "De datos brutos de supply chain a decisiones priorizadas · V2.0.13",
    "🔴 Critical": "🔴 Crítico", "🟠 Review": "🟠 Revisar", "🛒 Purchase need": "🛒 Necesidad de compra",
    "💰 Inventory": "💰 Inventario", "📈 Next month": "📈 Próximo mes",
    "Critical inventory exposure": "Exposición de inventario crítico",
    "Excess inventory": "Exceso de inventario", "Immediate actions": "Acciones inmediatas",
    "Service risk": "Riesgo de servicio", "Excess exposure": "Exposición de exceso",
    "Median PO cover": "Cobertura media de pedidos", "Top supplier risk": "Mayor riesgo de proveedor",
    "🔴 Explain critical SKUs": "🔴 Explicar SKUs críticos",
    "🛒 Explain purchase plan": "🛒 Explicar plan de compras",
    "⚠️ Explain service risk": "⚠️ Explicar riesgo de servicio",
    "💰 Inventory value": "💰 Valor de inventario", "🔄 Inventory turns": "🔄 Rotación de inventario",
    "📦 Days of cover": "📦 Días de cobertura", "🟢 Lead-time coverage": "🟢 Cobertura del lead time",
    "🔴 Critical SKUs": "🔴 SKUs críticos", "⚠️ Service risk": "⚠️ Riesgo de servicio",
    "🟡 Excess inventory": "🟡 Exceso de inventario", "🛒 Purchase requirement": "🛒 Necesidad de compra",
    "📨 Open PO value": "📨 Valor de pedidos abiertos", "🚚 Avg lead time": "🚚 Lead time medio",
    "🏭 Suppliers": "🏭 Proveedores", "📈 Next-month forecast": "📈 Forecast del próximo mes",
    "📊 Explain KPI health": "📊 Explicar salud de KPIs", "🚚 Supplier exposure": "🚚 Exposición por proveedor",
    "💰 Working capital": "💰 Capital circulante", "Purchase exposure": "Exposición de compras",
    "Risk addressed": "Riesgo abordado", "Blocked exposure": "Exposición bloqueada",
    "Explain the recommended action for": "Explicar la acción recomendada para",
    "Purchase need": "Necesidad de compra", "Required stock value": "Valor de stock requerido",
    "Rows": "Filas", "SKUs": "SKUs", "Suppliers": "Proveedores", "Warnings": "Avisos", "Critical": "Crítico",
    "Actions": "Acciones", "Immediate": "Inmediatas", "Purchase value": "Valor de compra",
    "Generate supplier communication": "Generar comunicación al proveedor",
    "🤖 Your Supply Chain Copilot": "🤖 Tu Supply Chain Copilot",
    "### ⚡ Quick analyses": "### ⚡ Análisis rápidos",
    "🛒 Purchase priorities": "🛒 Prioridades de compra", "💰 Reduce inventory": "💰 Reducir inventario",
    "🚚 Supplier risk": "🚚 Riesgo de proveedores", "📈 Demand outlook": "📈 Perspectiva de demanda",
    "🔄 What changed?": "🔄 ¿Qué ha cambiado?", "📊 Logistics KPIs": "📊 KPIs logísticos",
    "🤖 Ask your Supply Chain Copilot": "🤖 Pregunta a tu Supply Chain Copilot",
    "Examples: “What should I buy this week?”, “Where is my biggest stockout risk?”, “Which suppliers need attention?”": "Ejemplos: «¿Qué debería comprar esta semana?», «¿Dónde está mi mayor riesgo de rotura de stock?», «¿Qué proveedores requieren atención?»",
    "Ask a supply-chain question…": "Haz una pregunta de supply chain…",
    "📤 Reporting Center": "📤 Centro de informes",
    "Purchase requirement": "Necesidad de compra", "Actions": "Acciones",
    "### 1. Executive Report": "### 1. Informe ejecutivo",
    "Same executive KPIs and priorities, with editable tables, formatting and charts in Excel.": "Los mismos KPIs ejecutivos y prioridades, con tablas editables, formato y gráficos en Excel.",
    "The Excel workbook mirrors the detailed reports and adds a Change Monitor sheet.": "El libro de Excel replica los informes detallados y añade una hoja de Monitor de cambios.",
    "### 4. Planning Agent": "### 4. Agente de planificación",
    "### 5. Complete Management Pack": "### 5. Pack completo de gestión",
    "The Excel pack combines the executive dashboard, detailed report sheets, Change Monitor and normalized source data in one workbook.": "El pack de Excel combina el dashboard ejecutivo, las hojas de informes detallados, el Monitor de cambios y los datos fuente normalizados en un único libro.",
    "Raw CSV exports remain removed from the reporting workflow. HTML and Excel are now the primary shareable outputs.": "Las exportaciones CSV siguen fuera del flujo de reporting. HTML y Excel son ahora los principales formatos compartibles.",
    "HTML and Excel contain the same Change Monitor analysis: KPIs, period deltas, top changes, classification and reasons.": "HTML y Excel contienen el mismo análisis del Monitor de cambios: KPIs, variaciones por periodo, principales cambios, clasificación y motivos.",
    "▶️ Run prepared analysis": "▶️ Ejecutar análisis preparado", "✖️ Clear prepared analysis": "✖️ Limpiar análisis preparado",
    "📗 Export Decision Center to Excel": "📗 Exportar Centro de decisiones a Excel",
    "📗 Export Logistics KPI Dashboard to Excel": "📗 Exportar Dashboard de KPIs logísticos a Excel",
    "📊 Planning Agent Report (HTML)": "📊 Informe del Agente de planificación (HTML)",
    "📗 Planning Agent Report (Excel)": "📗 Informe del Agente de planificación (Excel)",
    "📗 Export Inventory to Excel": "📗 Exportar Inventario a Excel", "📗 Export Forecast to Excel": "📗 Exportar Forecast a Excel",
    "📗 Export ABC/XYZ to Excel": "📗 Exportar ABC/XYZ a Excel", "📗 Export Suppliers to Excel": "📗 Exportar Proveedores a Excel",
    "🌐 Data Quality Report (HTML)": "🌐 Informe de calidad de datos (HTML)", "📗 Data Quality Report (Excel)": "📗 Informe de calidad de datos (Excel)",
    "Action": "Acción", "Owner": "Responsable", "Deadline": "Fecha límite",
    "⬇️ Download supplier message": "⬇️ Descargar mensaje al proveedor",
    "🌐 Download Action Plan Report (HTML)": "🌐 Descargar informe del Plan de acción (HTML)",
    "📗 Download Action Plan Report (Excel)": "📗 Descargar informe del Plan de acción (Excel)",
    "Purchase requirement": "Necesidad de compra", "Action changes": "Cambios de acción",
    "📤 Export Change Monitor": "📤 Exportar Monitor de cambios",
    "📊 Download Change Monitor Report (HTML)": "📊 Descargar informe del Monitor de cambios (HTML)",
    "📗 Download Change Monitor Report (Excel)": "📗 Descargar informe del Monitor de cambios (Excel)",
    "Top worsened / watch": "Principales empeoramientos / vigilancia", "Top improved": "Principales mejoras",
    "Changes": "Cambios",
    "📊 Executive Report (HTML)": "📊 Informe ejecutivo (HTML)", "📗 Executive Report (Excel)": "📗 Informe ejecutivo (Excel)",
    "📊 Inventory & Service Risk (HTML)": "📊 Inventario y riesgo de servicio (HTML)",
    "🛒 Purchase Plan (HTML)": "🛒 Plan de compras (HTML)", "🚚 Supplier Risk (HTML)": "🚚 Riesgo de proveedores (HTML)",
    "📝 Weekly Action Plan (HTML)": "📝 Plan de acción semanal (HTML)", "🧹 Data Quality (HTML)": "🧹 Calidad de datos (HTML)",
    "📗 Detailed Reports (Excel)": "📗 Informes detallados (Excel)", "🔄 Change Monitor Report (HTML)": "🔄 Informe del Monitor de cambios (HTML)",
    "📦 Management Pack (ZIP)": "📦 Pack de gestión (ZIP)", "📗 Complete Management Pack (Excel)": "📗 Pack completo de gestión (Excel)",
    "There are more unfavorable changes": "Hay más cambios desfavorables", "than favorable changes": "que favorables",
    "The evolution is mostly favorable": "La evolución es mayoritariamente favorable", "improved": "mejorados",
    "versus": "frente a", "worsened": "empeorados",
    "The evolution is balanced between improvements and deteriorations.": "La evolución está equilibrada entre mejoras y empeoramientos.",
    "Running Supply Chain analysis...": "Ejecutando análisis de Supply Chain...",
    "Analyzing...": "Analizando...",
    "The MVP forecast uses a weighted average of the last 6 months plus a linear trend. The next iteration can add seasonality, intermittent demand and alternative models.": "El forecast del MVP utiliza una media ponderada de los últimos 6 meses más una tendencia lineal. La siguiente iteración puede añadir estacionalidad, demanda intermitente y modelos alternativos.",
    "HTML and Excel use the current filtered view. HTML includes KPIs and an executive presentation; Excel includes Summary, Action Plan and Supplier Summary with filters.": "HTML y Excel utilizan la vista filtrada actual. HTML incluye KPIs y una presentación ejecutiva; Excel incluye Summary, Action Plan y Supplier Summary con filtros.",
    "Need at least two historical periods to compare evolution.": "Se necesitan al menos dos periodos históricos para comparar la evolución.",
    "📦 Supply Chain AI Copilot V2.0.13 — recommendations require planner validation before execution.": "📦 Supply Chain AI Copilot V2.0.13 — las recomendaciones requieren validación del planner antes de su ejecución.",
    "Supply Chain AI Copilot V2.0.13 — recommendations require planner validation before execution.": "Supply Chain AI Copilot V2.0.13 — las recomendaciones requieren validación del planner antes de su ejecución.",
    "Safety stock floor": "Stock de seguridad mínimo", "Service level target": "Objetivo de nivel de servicio",
    "Language": "Idioma", "rows": "filas", "suppliers": "proveedores", "units": "unidades", "Fingerprint": "Huella",
    "Executive": "Ejecutivo", "Action Plan": "Plan de acción", "Data Quality": "Calidad de datos", "Inventory Risk": "Riesgo de inventario",
    "Purchase Plan": "Plan de compras", "Supplier Risk": "Riesgo de proveedores", "Planning Agent": "Agente de planificación",
    "Change Monitor": "Monitor de cambios", "Summary": "Resumen", "Quality Checks": "Comprobaciones de calidad", "Priorities": "Prioridades",
    "Purchase Plan": "Plan de compras", "Supplier Summary": "Resumen de proveedores", "KPI Catalogue": "Catálogo de KPIs",
    "Monthly Trend": "Tendencia mensual", "Supplier Exposure": "Exposición por proveedor", "ABC XYZ": "ABC XYZ", "Lead Time Profile": "Perfil de lead time",
    "Inventory Health": "Salud del inventario", "Demand Outlook": "Perspectiva de demanda", "Normalized Data": "Datos normalizados", "Source Data": "Datos fuente",
    "Excel export unavailable": "Exportación a Excel no disponible", "Prepared analysis": "Análisis preparado",
    "### Priorities for this week": "### Prioridades de esta semana",
    "| Priority | SKU | Action | Timing | Quantity | Risk | Confidence |": "| Prioridad | SKU | Acción | Timing | Cantidad | Riesgo | Confianza |",
    "Buy now": "Comprar ahora", "Confirm PO": "Confirmar PO", "Do not buy": "No comprar", "Review": "Revisar", "Monitor": "Monitorizar",
    "cover": "cobertura", "Total recommended purchase": "Compra recomendada total", "Current excess inventory": "Inventario actualmente en exceso",
    "**Immediate action:** issue/validate orders for `BUY_NOW` SKUs and confirm the delivery date with the supplier.": "**Acción inmediata:** emitir/validar pedidos de los SKUs `BUY_NOW` y confirmar la fecha de entrega con el proveedor.",
    "Explain the 5 most critical SKUs, what is driving the risk and which action should be reviewed first.": "Explica los 5 SKUs críticos más importantes, qué está provocando el riesgo y qué acción debería revisarse primero.",
    "Explain the current purchase plan, which suppliers concentrate the most value and what the purchase priorities are.": "Explica el plan de compras actual, qué proveedores concentran más valor y cuáles son las prioridades de compra.",
    "Explain where service risk is concentrated and which actions could reduce it without creating unnecessary purchases.": "Explica dónde está concentrado el riesgo de servicio y qué acciones podrían reducirlo sin generar compras innecesarias.",
    "Analyze the health of the main logistics KPIs, identify the signals that require attention and explain their causes.": "Analiza la salud de los principales KPIs logísticos, identifica las señales que requieren atención y explica sus causas.",
    "Analyze supplier exposure and tell me where the planner should focus attention.": "Analiza la exposición por proveedor y dime dónde debería concentrar la atención del planner.",
    "Analyze inventory, excess, coverage and recommended purchases from a working-capital perspective.": "Analiza inventario, exceso, cobertura y compras recomendadas desde la perspectiva de capital circulante.",
    "What should I buy this week and what are the 3 most important priorities?": "¿Qué debería comprar esta semana y cuáles son las 3 prioridades más importantes?",
    "Where can I reduce inventory without materially increasing service risk?": "¿Dónde puedo reducir inventario sin aumentar demasiado el riesgo de servicio?",
    "Which suppliers require the most attention and why?": "¿Qué proveedores requieren más atención y por qué?",
    "What demand changes could alter my purchase decisions?": "¿Qué cambios de demanda pueden cambiar mis decisiones de compra?",
    "What changed between the latest period and the previous one, and what are the 3 largest variations?": "¿Qué ha cambiado entre el último periodo y el anterior y cuáles son las 3 mayores variaciones?",
    "What is the state of the main logistics KPIs and which ones require attention?": "¿Cuál es el estado de los principales KPI logísticos y cuáles requieren atención?",
    "Comparing": "Comparando",
    "## Supply Chain Agent — analysis": "## Supply Chain Agent — análisis",
    "turns": "rotación", "aggregate cover": "cobertura agregada", "lead-time coverage": "cobertura de lead time",
    "excess": "exceso", "open POs": "PO abiertas", "recommended purchase": "compra recomendada",
    "Recommended purchase": "Compra recomendada", "lines": "líneas", "critical": "críticas", "units": "uds",
    "Estimated excess": "Exceso estimado", "across": "en", "Service exposure": "Exposición de servicio",
    "service risk lower": "riesgo de servicio", "Largest demand increases:": "Mayores subidas de demanda:",
    "forecast": "forecast", "Execution plan": "Plan de ejecución", "immediate actions": "acciones inmediatas",
    "purchase exposure": "exposición de compra", "service risk potentially addressable": "riesgo de servicio potencialmente abordable",
    "Step": "Paso", "No comparable periods are available.": "No hay periodos comparables disponibles.",
    "Comparison": "Comparación", "action changes": "cambios de acción", "coverage": "cobertura",
    "OpenAI authentication failed": "Autenticación de OpenAI fallida", "OpenAI rate limit reached": "Límite/cuota de OpenAI alcanzado",
    "OpenAI error": "Error de OpenAI", "Agent error": "Error del agente", "unknown": "desconocido",
})


# Translate exact user-facing labels automatically; complex HTML/Markdown is left untouched unless it is a known phrase.
for _method_name in [
    "title", "header", "subheader", "caption", "write", "info", "warning", "error", "success",
    "button", "download_button", "selectbox", "multiselect", "radio", "slider", "number_input",
    "text_input", "text_area", "file_uploader", "chat_input", "expander", "markdown", "spinner"
]:
    _original_method = getattr(st, _method_name, None)
    if _original_method is not None and not getattr(_original_method, "_sc_i18n_wrapped", False):
        def _make_i18n_wrapper(_fn):
            def _wrapped(label, *args, **kwargs):
                return _fn(tr(label), *args, **kwargs)
            _wrapped._sc_i18n_wrapped = True
            return _wrapped
        setattr(st, _method_name, _make_i18n_wrapper(_original_method))

# Display labels for dataframes. Calculations always keep the canonical English field names.
_COLUMN_TRANSLATIONS_ES = {
    "SKU":"SKU", "Description":"Descripción", "Supplier":"Proveedor", "Status":"Estado",
    "Action":"Acción", "Action_Timing":"Momento de acción", "Decision_Confidence":"Confianza de decisión",
    "Days_Cover":"Días de cobertura", "Lead_Time_Days":"Lead time (días)", "Recommended_Order":"Pedido recomendado",
    "Purchase_Value":"Valor de compra", "Decision_Score":"Puntuación de decisión", "Inventory_Value":"Valor de inventario",
    "Safety_Stock":"Stock de seguridad", "ABC_XYZ":"ABC/XYZ", "Annual_Sales":"Ventas anuales",
    "Avg_Monthly_Demand":"Demanda mensual media", "Forecast_Next_Month":"Forecast próximo mes",
    "Forecast_Change_Pct":"Cambio forecast %", "Trend_Units_Per_Month":"Tendencia unidades/mes", "Demand_CV":"CV de demanda",
    "Annual_Consumption_Value":"Valor consumo anual", "ABC":"ABC", "XYZ":"XYZ", "SKUs":"SKUs",
    "Critical":"Críticos", "Avg_Cover":"Cobertura media", "Purchase_Value":"Valor de compra",
    "Priority":"Prioridad", "Owner":"Responsable", "Deadline":"Fecha límite", "Reason":"Motivo",
    "Confidence":"Confianza", "Purchase":"Compra", "Previous_Action":"Acción anterior", "Current_Action":"Acción actual",
    "Action_Transition":"Transición de acción", "Previous_Days_Cover":"Cobertura anterior", "Current_Days_Cover":"Cobertura actual",
    "Days_Cover_Delta":"Delta cobertura", "Purchase_Value_Delta":"Delta valor compra", "Service_Risk_Delta":"Delta riesgo servicio",
    "Excess_Value_Delta":"Delta exceso", "Sales_Delta_Pct":"Delta ventas %", "Change_Classification":"Clasificación del cambio",
    "Change_Reason":"Motivo del cambio", "Category":"Categoría", "Check":"Comprobación", "Count":"Cantidad",
    "Details":"Detalles", "Status":"Estado", "Year":"Año", "Month":"Mes", "Sales":"Ventas", "Stock":"Stock",
    "Open_PO":"Pedidos abiertos", "MOQ":"MOQ", "Unit_Cost":"Coste unitario", "Inventory_Value":"Valor inventario",
}

def localize_df(df):
    if st.session_state.get("language", "English") != "Spanish" or df is None:
        return df
    out = df.copy()
    out.columns = [_COLUMN_TRANSLATIONS_ES.get(str(c), str(c)) for c in out.columns]
    value_maps = {
        "Status": {"🔴 CRITICAL":"🔴 CRÍTICO", "🟠 REVIEW":"🟠 REVISAR", "🟡 EXCESS":"🟡 EXCESO", "🟢 OK":"🟢 OK"},
        "Action": {"BUY_NOW":"COMPRAR AHORA", "CONFIRM_PO":"CONFIRMAR PO", "DO_NOT_BUY":"NO COMPRAR", "REVIEW":"REVISAR", "MONITOR":"MONITORIZAR", "BUY NOW":"COMPRAR", "BUY":"COMPRAR", "CONFIRM OPEN PO":"CONFIRMAR PO", "REVIEW REPLENISHMENT POLICY":"REVISAR POLÍTICA DE REPOSICIÓN", "BLOCK NEW REPLENISHMENT":"BLOQUEAR NUEVA REPOSICIÓN"},
        "Owner": {"Planner":"Planner", "Planner / Buyer":"Planner / Comprador"},
        "Deadline": {"Today":"Hoy", "This week":"Esta semana", "Next cycle":"Próximo ciclo", "Immediate":"Inmediato", "Routine":"Rutina"},
        "Change_Classification": {"WORSENED":"EMPEORADO", "IMPROVED":"MEJORADO", "WATCH":"VIGILAR", "STABLE":"ESTABLE"},
        "Category": {"Completeness":"Completitud", "Validity":"Validez", "Consistency":"Consistencia", "Uniqueness":"Unicidad"},
        "Check": {"Missing values":"Valores ausentes", "Negative values":"Valores negativos", "Month range":"Rango de meses", "Duplicate SKU-period":"SKU-periodo duplicado", "Blank SKU":"SKU vacío", "Blank Supplier":"Proveedor vacío"},
        "Action_Timing": {"Immediate":"Inmediato", "Today":"Hoy", "This week":"Esta semana", "Next cycle":"Próximo ciclo", "Routine":"Rutina", "Monitor":"Monitorizar"},
    }
    for canonical_col, mapping in value_maps.items():
        display_col = _COLUMN_TRANSLATIONS_ES.get(canonical_col, canonical_col)
        if canonical_col in df.columns and display_col in out.columns:
            def _map_value(v):
                sv = str(v)
                if canonical_col == "Action":
                    for en, es in mapping.items():
                        if sv.startswith(en):
                            return sv.replace(en, es, 1)
                return mapping.get(sv, v)
            out[display_col] = out[display_col].map(_map_value)
    return out

# Keep calculations in canonical English while presenting tables and common controls in the selected language.
_original_st_dataframe = st.dataframe
def _localized_dataframe(data, *args, **kwargs):
    return _original_st_dataframe(localize_df(data), *args, **kwargs)
st.dataframe = _localized_dataframe

_original_st_metric = st.metric
def _localized_metric(label, *args, **kwargs):
    return _original_st_metric(tr(label), *args, **kwargs)
st.metric = _localized_metric

# -----------------------------
# User-friendly UI layer
# -----------------------------
st.markdown("""
<style>
/* Supply Chain AI Copilot — usability layer */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 12px 14px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] { font-size: 0.82rem; }
[data-testid="stMetricValue"] { font-size: 1.45rem; }
div[data-testid="stTabs"] button { font-weight: 650; }
div[data-testid="stExpander"] { border-radius: 12px; }
.sc-workflow {
    border: 1px solid #dbe4ee; background: #f8fafc; border-radius: 14px;
    padding: 14px 16px; margin: 8px 0 18px 0;
}
.sc-step { display:inline-block; margin-right:22px; font-weight:650; }
.sc-muted { color:#64748b; font-size:0.86rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📦 Supply Chain AI")
    st.caption(tr("Decision Intelligence for planners · V2.0.13"))

    language_choice = st.selectbox(f"🌐 {tr('Language')}", ["English", "Español"], index=0 if st.session_state.language == "English" else 1, key="language_selector")
    st.session_state.language = "English" if language_choice == "English" else "Spanish"

    uploaded = st.file_uploader(
        tr("Historical demand and inventory"),
        type=["csv", "xlsx", "xls"],
        key="historical_demand_inventory_uploader",
        help=tr("Upload historical demand and inventory data for analysis. CSV and Excel are supported.")
    )
    safety_days = st.slider(tr("Safety stock floor (days)"), 0, 90, 10)
    service = st.select_slider(tr("Service level"), options=[0.90,0.95,0.975,0.99], value=0.95)
    st.subheader(tr("🤖 OpenAI"))

    secret_key = ""
    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    stored_key = secret_key or env_key

    source = (
        "Streamlit Secrets" if secret_key
        else ("Environment variable" if env_key else "None")
    )

    model = st.selectbox(
        tr("Model"),
        ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        index=0,
        help=tr("Current models available in the Responses API.")
    )


    if stored_key:
        prefix = stored_key[:8] if len(stored_key) >= 8 else stored_key
        suffix = stored_key[-4:] if len(stored_key) >= 4 else ""
        st.success(tr(f"API key detected · {source}"))
        st.caption(f"{tr('Fingerprint')}: `{prefix}…{suffix}`")
    else:
        st.warning(tr("No OPENAI_API_KEY configured. Local mode will be used."))

    use_manual = st.checkbox(
        tr("Test another key for this session only"),
        value=False
    )

    manual_key = ""
    if use_manual:
        manual_key = st.text_input(
            tr("Manual API key"),
            type="password"
        ).strip()

    effective_key = manual_key if use_manual and manual_key else stored_key

    if st.button(tr("🔌 Diagnose OpenAI connection"), use_container_width=True):
        ok, message, details = test_openai_connection(effective_key, model)

        if ok:
            st.success(f"{tr('✅ OpenAI connected')}: {message}")
        else:
            st.error(tr("❌ Unable to validate the connection."))
            if details:
                st.code(
                    f"HTTP: {details.get('http_status')}\n"
                    f"Code: {details.get('error_code')}\n"
                    f"Type: {details.get('error_type')}\n"
                    f"Exception: {details.get('exception')}\n"
                    f"Message: {message}",
                    language="text"
                )
            else:
                st.code(message, language="text")

    st.caption(
        tr("The API key is never displayed in full or stored in GitHub.")
    )

    if st.button(tr("🔄 Load demo")):
        st.session_state.raw_data = sample_data()
        st.session_state.analysis = analyze(st.session_state.raw_data, safety_days, service)
        st.session_state.chat = []
        st.rerun()



def _excel_tab_export_bytes(title, sheets, kpis=None):
    """Create a polished Excel workbook for an individual application tab."""
    if xlsxwriter is None:
        return None
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        title_fmt = wb.add_format({"bold": True, "font_size": 18, "font_color": "#17365D"})
        subtitle_fmt = wb.add_format({"italic": True, "font_color": "#666666"})
        header_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D", "border": 0, "text_wrap": True, "valign": "vcenter"})
        currency_fmt = wb.add_format({"num_format": '€#,##0.00', "valign": "top"})
        integer_fmt = wb.add_format({"num_format": '#,##0', "valign": "top"})
        number_fmt = wb.add_format({"num_format": '#,##0.00', "valign": "top"})

        # Executive summary sheet
        summary = wb.add_worksheet(tr("Summary"))
        summary.hide_gridlines(2)
        summary.write(0, 0, title, title_fmt)
        summary.write(1, 0, "Exported from Supply Chain AI Copilot V2.0.13", subtitle_fmt)
        if kpis:
            summary.write(3, 0, "Key metrics", header_fmt)
            for i, (label, value) in enumerate(kpis.items(), start=4):
                summary.write(i, 0, label, header_fmt)
                summary.write(i, 1, value)
            summary.set_column(0, 0, 28)
            summary.set_column(1, 1, 22)

        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31]
            out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            out.to_excel(writer, sheet_name=safe_name, index=False, startrow=2)
            ws = writer.sheets[safe_name]
            ws.hide_gridlines(2)
            ws.write(0, 0, f"{title} — {safe_name}", title_fmt)
            ws.write(1, 0, "Decision-ready table · filters enabled · source: current application view", subtitle_fmt)
            ws.freeze_panes(3, 0)
            if len(out.columns):
                ws.autofilter(2, 0, 2 + max(len(out), 1), len(out.columns)-1)
            for j, col in enumerate(out.columns):
                col_lower = str(col).lower()
                width = min(max(len(str(col)) + 2, 12), 32)
                if col_lower in {"description", "reason", "change_reason", "execution_task", "dependency", "planning_rationale"}:
                    width = 34
                ws.set_column(j, j, width)
                # Apply sensible formats to numeric/currency columns.
                if any(k in col_lower for k in ["value", "cost", "exposure", "risk"]):
                    ws.set_column(j, j, width, currency_fmt)
                elif any(k in col_lower for k in ["qty", "order", "sales", "stock", "po", "skus", "units", "priority", "critical", "review"]):
                    ws.set_column(j, j, width, integer_fmt)
                elif any(k in col_lower for k in ["pct", "cv", "cover", "score", "delta", "lead"]):
                    ws.set_column(j, j, width, number_fmt)
            # Format header row consistently after pandas writes it.
            for j, col in enumerate(out.columns):
                ws.write(2, j, col, header_fmt)

    return buf.getvalue()

# -----------------------------
# Load data
# -----------------------------
if "raw_data" not in st.session_state:
    st.session_state.raw_data = sample_data()

if uploaded:
    raw = normalize_columns(read_uploaded(uploaded))
    valid, error = validate(raw)
    if not valid:
        st.error(error)
        st.stop()
    st.session_state.raw_data = raw
    st.session_state.analysis = analyze(raw, safety_days, service)

if st.session_state.analysis is None:
    st.session_state.raw_data = st.session_state.raw_data.copy()
    st.session_state.analysis = analyze(st.session_state.raw_data, safety_days, service)

raw = st.session_state.raw_data
a = st.session_state.analysis

# Schema guard: Streamlit can keep session_state across code updates.
# If the analysis was calculated by an older version, rebuild it so newly
# introduced decision columns are present.
_REQUIRED_ANALYSIS_COLUMNS = {
    "Service_Risk_Value",
    "Excess_Inventory_Value",
    "Open_PO_Cover_Days",
    "PO_Adequacy",
    "Action",
    "Decision_Confidence",
    "Forecast_Change_Pct",
}
if not _REQUIRED_ANALYSIS_COLUMNS.issubset(set(a.columns)):
    a = analyze(raw, safety_days, service)
    st.session_state.analysis = a

# ------------------------------------------------------------------
# Global operational filters
# ------------------------------------------------------------------
# The source dataset remains untouched. Filters only affect operational
# decision views, while Data Quality continues to inspect the full source.
with st.sidebar:
    with st.expander(f"🔎 {tr('Global filters')}", expanded=False):
        st.caption(tr("Filter the operational views without changing the source dataset."))

        if st.button(tr("Reset filters"), use_container_width=True, key="reset_global_filters"):
            for _k, _v in {
                "global_sku_search": "",
                "global_suppliers": [],
                "global_statuses": [],
                "global_abc": [],
                "global_xyz": [],
            }.items():
                st.session_state[_k] = _v
            st.rerun()

        st.text_input(
            tr("SKU / description search"),
            key="global_sku_search",
            placeholder="e.g. SKU-001 or pump",
            help=tr("Search by SKU or product description.")
        )

        supplier_options = sorted(a["Supplier"].dropna().astype(str).unique().tolist())
        st.multiselect(
            tr("Supplier"),
            supplier_options,
            key="global_suppliers",
            placeholder=tr("All suppliers")
        )

        st.multiselect(
            tr("Inventory status"),
            status_display_options(),
            key="global_statuses",
            placeholder=tr("All statuses")
        )
        st.multiselect(
            tr("ABC class"),
            ["A", "B", "C"],
            key="global_abc",
            placeholder=tr("All ABC classes")
        )
        st.multiselect(
            tr("XYZ class"),
            ["X", "Y", "Z"],
            key="global_xyz",
            placeholder=tr("All XYZ classes")
        )

_search = str(st.session_state.get("global_sku_search", "")).strip().lower()
_selected_suppliers = set(st.session_state.get("global_suppliers", []))
_selected_statuses = {status_from_display(x) for x in st.session_state.get("global_statuses", [])}
_selected_abc = set(st.session_state.get("global_abc", []))
_selected_xyz = set(st.session_state.get("global_xyz", []))

_filter_mask = pd.Series(True, index=a.index)
if _search:
    _filter_mask &= (
        a["SKU"].astype(str).str.lower().str.contains(_search, regex=False, na=False)
        | a["Description"].astype(str).str.lower().str.contains(_search, regex=False, na=False)
    )
if _selected_suppliers:
    _filter_mask &= a["Supplier"].astype(str).isin(_selected_suppliers)
if _selected_statuses:
    _filter_mask &= a["Status"].astype(str).isin(_selected_statuses)
if _selected_abc:
    _filter_mask &= a["ABC"].astype(str).isin(_selected_abc)
if _selected_xyz:
    _filter_mask &= a["XYZ"].astype(str).isin(_selected_xyz)

_a_source = a
a = a.loc[_filter_mask].copy()
_filtered_skus = set(a["SKU"].astype(str))
raw_view = raw[raw["SKU"].astype(str).isin(_filtered_skus)].copy()

with st.sidebar:
    if _filter_mask.all():
        st.caption(f"{tr('Active filters')}: {tr('None')}" if tr('None') != 'None' else f"{tr('Active filters')}: None")
    else:
        st.caption(f"{tr('Active filters')}: {a['SKU'].nunique():,} {tr('of')} {_a_source['SKU'].nunique():,} {tr('filtered SKUs')}")

K = kpis(a)
logistics_dashboard = build_logistics_dashboard(a, raw_view)
dq = data_quality_report(raw)
plan = build_action_plan(a)
planning, planning_meta = build_planning_agent(a)

available_periods = _periods_from_raw(raw)
if len(available_periods) >= 2:
    default_current = st.session_state.get("change_current_period", available_periods[-1])
    if default_current not in available_periods:
        default_current = available_periods[-1]
    current_idx = available_periods.index(default_current)
    prev_options = available_periods[:current_idx]
    if not prev_options:
        prev_options = available_periods[:-1]
    default_previous = st.session_state.get(
        "change_previous_period",
        prev_options[-1] if prev_options else available_periods[-2]
    )
    if default_previous not in prev_options:
        default_previous = prev_options[-1]
    comparison, comparison_current_a, comparison_previous_a, comparison_meta = build_period_comparison(
        raw, safety_days, service, default_current, default_previous
    )
else:
    default_current = default_previous = None
    comparison = pd.DataFrame()
    comparison_current_a = comparison_previous_a = None
    comparison_meta = {"available_periods": available_periods, "has_comparison": False}

# -----------------------------
# Comparison controls
# -----------------------------
with st.sidebar:
    st.markdown(f"### 📁 {tr('Current dataset')}")
    st.caption(f"{len(raw):,} {tr('rows')} · {a['SKU'].nunique():,} SKUs · {a['Supplier'].nunique():,} {tr('suppliers')}")
    if available_periods:
        st.caption(f"{tr('Period')}: **{available_periods[0]} → {available_periods[-1]}**")
    if K["critical"] > 0:
        st.warning(f"{K['critical']:,} {tr('critical SKUs require attention.')}")
    else:
        st.success(tr("No critical SKUs under the current planning parameters."))

    with st.expander(tr("⚙️ Planning assumptions")):
        st.caption(f"{tr('Safety stock floor')}: **{safety_days} {tr('days')}**")
        st.caption(f"{tr('Service level target')}: **{service:.1%}**")
        st.caption(tr("These parameters affect safety stock, coverage and purchase recommendations."))

    if len(available_periods) >= 2:
        with st.expander(tr("🔄 Period comparison"), expanded=False):
            selected_current = st.selectbox(
                tr("Current period"),
                available_periods,
                index=available_periods.index(default_current),
                key="change_current_period"
            )
            current_idx = available_periods.index(selected_current)
            prev_options = available_periods[:current_idx] or [available_periods[0]]
            selected_previous = st.selectbox(
                tr("Compare with"),
                prev_options,
                index=prev_options.index(default_previous) if default_previous in prev_options else len(prev_options)-1,
                key="change_previous_period"
            )
            st.caption(f"{tr('Current')}: {selected_current} · {tr('Previous')}: {selected_previous}")

# Add/refresh forecast change vs historical monthly average
a["Forecast_Change_Pct"] = np.where(
    a["Avg_Monthly_Demand"] > 0,
    (a["Forecast_Next_Month"]/a["Avg_Monthly_Demand"]-1)*100,
    0
)

# -----------------------------
# Header
# -----------------------------
st.title("📦 Supply Chain AI Copilot")
st.caption(tr("From raw supply-chain data to prioritized decisions · V2.0.13"))

if not _filter_mask.any():
    st.warning(tr("No SKUs match the selected filters."))
else:
    _active_filter_parts = []
    if _search:
        _active_filter_parts.append(f"{tr('Search')}: {st.session_state.get('global_sku_search')}")
    if _selected_suppliers:
        _active_filter_parts.append(f"{tr('Supplier')}: {len(_selected_suppliers)}")
    if _selected_statuses:
        _active_filter_parts.append(f"{tr('Inventory status')}: {len(_selected_statuses)}")
    if _selected_abc:
        _active_filter_parts.append(f"ABC: {', '.join(sorted(_selected_abc))}")
    if _selected_xyz:
        _active_filter_parts.append(f"XYZ: {', '.join(sorted(_selected_xyz))}")
    if _active_filter_parts:
        st.info(f"🔎 {tr('Active filters')}: " + " · ".join(_active_filter_parts))

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SKUs", K["sku"])
c2.metric("🔴 Critical", K["critical"])
c3.metric("🟠 Review", K["review"])
c4.metric("🛒 Purchase need", f"€{K['purchase']:,.0f}")
c5.metric("💰 Inventory", f"€{K['inventory']:,.0f}")
c6.metric("📈 Next month", f"{K['forecast']:,.0f}")



tabs = st.tabs([
    "🎯 Decision Center" if st.session_state.language == "English" else "🎯 Centro de decisiones",
    "📊 Logistics Dashboard" if st.session_state.language == "English" else "📊 Dashboard logístico",
    "🧠 Planning Agent" if st.session_state.language == "English" else "🧠 Agente de planificación",
    "📊 Inventory" if st.session_state.language == "English" else "📊 Inventario",
    "📈 Forecast",
    "🧩 ABC/XYZ",
    "🚚 Suppliers" if st.session_state.language == "English" else "🚚 Proveedores",
    "🧪 Scenarios" if st.session_state.language == "English" else "🧪 Escenarios",
    "🏭 MRP" if st.session_state.language == "English" else "🏭 MRP",
    "🧹 Data Quality" if st.session_state.language == "English" else "🧹 Calidad de datos",
    "📝 Action Plan" if st.session_state.language == "English" else "📝 Plan de acción",
    "🔄 Change Monitor" if st.session_state.language == "English" else "🔄 Monitor de cambios",
    "🤖 Copilot",
    "📤 Export" if st.session_state.language == "English" else "📤 Exportar"
])

# -----------------------------
# Decision Center
# -----------------------------
with tabs[0]:
    st.subheader(tr("What should the planner do now?"))
    st.markdown(decision_text(a))
    st.divider()

    top = a.sort_values("Decision_Score", ascending=False).head(15)
    st.dataframe(
        top[[
            "SKU","Description","Supplier","Status","Action","Action_Timing",
            "Decision_Confidence","Days_Cover","Lead_Time_Days",
            "Recommended_Order","Purchase_Value","Decision_Score"
        ]],
        use_container_width=True, hide_index=True
    )

    critical_value = float(a.loc[a["Action"]=="BUY_NOW", "Inventory_Value"].sum())
    excess_value = float(a.loc[a["Action"]=="DO_NOT_BUY", "Inventory_Value"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Critical inventory exposure", f"€{critical_value:,.0f}")
    m2.metric("Excess inventory", f"€{excess_value:,.0f}")
    m3.metric("Immediate actions", int((a["Action"].isin(["BUY_NOW","CONFIRM_PO"])).sum()))

    st.subheader(tr("Why these actions?"))
    st.info(tr("The engine treats a SKU as BUY_NOW when on-hand stock is below lead-time demand. REVIEW/CONFIRM_PO cases are handled separately to avoid double ordering when an open PO already exists. DO_NOT_BUY cases are flagged when coverage is materially above the policy threshold."))

    c1, c2, c3, c4 = st.columns(4)
    service_risk_value = float(a.get("Service_Risk_Value", pd.Series(0, index=a.index)).sum())
    excess_value = float(a.get("Excess_Inventory_Value", pd.Series(0, index=a.index)).sum())
    c1.metric("Service risk", f"€{service_risk_value:,.0f}")
    c2.metric("Excess exposure", f"€{excess_value:,.0f}")
    po_cover = a.get("Open_PO_Cover_Days", pd.Series(np.nan, index=a.index)).replace([np.inf, -np.inf], np.nan)
    c3.metric("Median PO cover", f"{po_cover.median():.1f}d" if po_cover.notna().any() else "—")
    supplier_tool = agent_supplier_tool(a)
    top_supplier = supplier_tool["rows"][0]["Supplier"] if supplier_tool["rows"] else "—"
    c4.metric("Top supplier risk", str(top_supplier))

    st.subheader(tr("Purchase plan by supplier"))
    po = export_purchase(a)
    if po.empty:
        st.success(tr("No purchase orders recommended."))
    else:
        st.dataframe(po, use_container_width=True, hide_index=True)

    st.markdown(f"#### 🤖 {tr('Continue with Copilot')}")
    dcq1, dcq2, dcq3 = st.columns(3)
    if dcq1.button("🔴 Explain critical SKUs", use_container_width=True, key="dc_critical_copilot"):
        st.session_state.copilot_prefill = tr("Explain the 5 most critical SKUs, what is driving the risk and which action should be reviewed first.")
    if dcq2.button("🛒 Explain purchase plan", use_container_width=True, key="dc_purchase_copilot"):
        st.session_state.copilot_prefill = tr("Explain the current purchase plan, which suppliers concentrate the most value and what the purchase priorities are.")
    if dcq3.button("⚠️ Explain service risk", use_container_width=True, key="dc_service_copilot"):
        st.session_state.copilot_prefill = tr("Explain where service risk is concentrated and which actions could reduce it without creating unnecessary purchases.")
    if st.session_state.get("copilot_prefill"):
        st.info(tr("Prepared question for Copilot. Go to the 🤖 Copilot tab to run it."))

    decision_export = _excel_tab_export_bytes(
        "Decision Center",
        {
            "Priorities": top[[
                "SKU","Description","Supplier","Status","Action","Action_Timing",
                "Decision_Confidence","Days_Cover","Lead_Time_Days",
                "Recommended_Order","Purchase_Value","Decision_Score"
            ]],
            "Purchase Plan": po if not po.empty else pd.DataFrame(),
        },
        {
            "SKUs": int(len(a)),
            "Service risk": float(service_risk_value),
            "Excess exposure": float(excess_value),
            "Immediate actions": int((a["Action"].isin(["BUY_NOW","CONFIRM_PO"])).sum()),
        }
    )
    if decision_export:
        st.download_button(
            "📗 Export Decision Center to Excel", decision_export,
            "decision_center_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key="decision_center_excel"
        )

# -----------------------------
# Logistics Dashboard
# -----------------------------
with tabs[1]:
    st.subheader(f"📊 {tr('Logistics KPI Dashboard')}")
    st.caption(tr("Executive view of inventory, service exposure, replenishment, supplier exposure and logistics efficiency."))
    d = logistics_dashboard
    m = d["meta"]

    # Executive KPI cards
    r1 = st.columns(4)
    r1[0].metric("💰 Inventory value", f"€{m['inventory_value']:,.0f}")
    r1[1].metric("🔄 Inventory turns", f"{m['inventory_turns']:.2f}x" if np.isfinite(m['inventory_turns']) else "—")
    r1[2].metric("📦 Days of cover", f"{m['aggregate_cover']:.1f}d" if np.isfinite(m['aggregate_cover']) else "—")
    r1[3].metric("🟢 Lead-time coverage", f"{m['lead_time_coverage_pct']:.1f}%")

    r2 = st.columns(4)
    r2[0].metric("🔴 Critical SKUs", f"{K['critical']}", f"{m['critical_pct']:.1f}% of SKUs")
    r2[1].metric("⚠️ Service risk", f"€{m['service_risk_value']:,.0f}", f"{(m['service_risk_value']/m['inventory_value']*100):.1f}% of inventory" if m['inventory_value'] else "—")
    r2[2].metric("🟡 Excess inventory", f"€{m['excess_value']:,.0f}", f"{m['excess_pct']:.1f}% of inventory")
    r2[3].metric("🛒 Purchase requirement", f"€{m['purchase_value']:,.0f}", f"{m['purchase_to_inventory_pct']:.1f}% of inventory" if m['inventory_value'] else "—")

    r3 = st.columns(4)
    r3[0].metric("📨 Open PO value", f"€{m['open_po_value']:,.0f}")
    r3[1].metric("🚚 Avg lead time", f"{m['avg_lead_time']:.1f}d")
    r3[2].metric("🏭 Suppliers", int(a["Supplier"].nunique()))
    r3[3].metric(tr("📈 Next-month forecast"), f"{a['Forecast_Next_Month'].sum():,.0f} {tr('units')}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### 📈 {tr('Sales value vs inventory value')}")
        if not d["trend"].empty:
            trend_chart = d["trend"].set_index("Period")[["Sales_Value", "Inventory_Value"]]
            st.line_chart(trend_chart, use_container_width=True)
        else:
            st.info(tr("No monthly history available for the trend chart."))
    with c2:
        st.markdown(f"#### 📦 {tr(tr("Inventory health"))}")
        status_chart = d["status"].set_index("Status")[["SKUs"]]
        st.bar_chart(status_chart, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"#### 🚚 {tr(tr("Supplier exposure"))}")
        supplier_chart = d["supplier"].head(10).set_index("Supplier")[["Inventory_Value", "Service_Risk_Value", "Purchase_Value"]]
        st.bar_chart(supplier_chart, use_container_width=True)
    with c4:
        st.markdown(f"#### ⏱️ {tr('Lead-time profile')}")
        lead_chart = d["lead_bins"].set_index("Lead_Time_Bucket")[["SKUs"]]
        st.bar_chart(lead_chart, use_container_width=True)

    st.markdown(f"#### 🧩 {tr('ABC / XYZ portfolio')}")
    abc_display = d["abc_xyz"].set_index("ABC_Class")
    st.dataframe(abc_display, use_container_width=True)

    st.markdown(f"#### 📋 {tr('Logistics KPI catalogue')}")
    kpi_display = d["kpis"].copy()
    kpi_display["Value"] = kpi_display.apply(
        lambda row: (
            f"€{row['Value']:,.0f}" if row["Unit"] == "€" else
            f"{row['Value']:.2f}x" if row["Unit"] == "x" else
            f"{row['Value']:.1f}d" if row["Unit"] == "days" else
            f"{row['Value']:.1f}%" if row["Unit"] == "%" else
            f"{row['Value']:,.0f}" if isinstance(row["Value"], (int, float, np.integer, np.floating)) else str(row["Value"])
        ), axis=1
    )
    st.dataframe(kpi_display, use_container_width=True, hide_index=True)

    st.markdown(f"#### 🔎 {tr('What the dashboard is telling the planner')}")
    insights = []
    if m["lead_time_coverage_pct"] < 90:
        insights.append(f"**Service exposure:** only {m['lead_time_coverage_pct']:.1f}% of SKUs currently cover lead-time demand.")
    else:
        insights.append(f"**Service coverage:** {m['lead_time_coverage_pct']:.1f}% of SKUs cover lead-time demand.")
    if m["excess_pct"] >= 20:
        insights.append(f"**Working capital:** excess inventory represents {m['excess_pct']:.1f}% of current inventory value.")
    if m["purchase_to_inventory_pct"] >= 20:
        insights.append(f"**Replenishment pressure:** recommended purchases equal {m['purchase_to_inventory_pct']:.1f}% of current inventory value.")
    if m["avg_lead_time"] > 30:
        insights.append(f"**Lead-time exposure:** average supplier lead time is {m['avg_lead_time']:.1f} days.")
    if not insights:
        insights.append("The current portfolio has no major threshold breach under the selected planning parameters.")
    for insight in insights:
        st.info(insight)

    st.markdown(f"#### 🤖 {tr('Investigate with Copilot')}")
    kq1, kq2, kq3 = st.columns(3)
    if kq1.button("📊 Explain KPI health", use_container_width=True, key="kpi_health_copilot"):
        st.session_state.copilot_prefill = tr("Analyze the health of the main logistics KPIs, identify the signals that require attention and explain their causes.")
    if kq2.button("🚚 Supplier exposure", use_container_width=True, key="kpi_supplier_copilot"):
        st.session_state.copilot_prefill = tr("Analyze supplier exposure and tell me where the planner should focus attention.")
    if kq3.button("💰 Working capital", use_container_width=True, key="kpi_wc_copilot"):
        st.session_state.copilot_prefill = tr("Analyze inventory, excess, coverage and recommended purchases from a working-capital perspective.")
    if st.session_state.get("copilot_prefill"):
        st.info(tr("Prepared question for Copilot. Go to the 🤖 Copilot tab to run it."))

    dashboard_export = _excel_tab_export_bytes(
        "Logistics KPI Dashboard",
        {
            "KPI Catalogue": d["kpis"],
            "Monthly Trend": d["trend"],
            "Supplier Exposure": d["supplier"],
            "ABC XYZ": d["abc_xyz"],
            "Lead Time Profile": d["lead_bins"],
            "Inventory Health": d["status"],
        },
        {
            "Inventory value": m["inventory_value"],
            "Inventory turns": m["inventory_turns"] if np.isfinite(m["inventory_turns"]) else "—",
            "Days of cover": m["aggregate_cover"] if np.isfinite(m["aggregate_cover"]) else "—",
            "Lead-time coverage %": m["lead_time_coverage_pct"],
            "Service risk": m["service_risk_value"],
            "Excess inventory": m["excess_value"],
            "Purchase requirement": m["purchase_value"],
            "Open PO value": m["open_po_value"],
        }
    )
    if dashboard_export:
        st.download_button(
            "📗 Export Logistics KPI Dashboard to Excel",
            dashboard_export,
            "logistics_kpi_dashboard.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="logistics_dashboard_excel"
        )
    st.caption(tr("Service and coverage KPIs are planning proxies derived from inventory, demand and lead-time data; they are not OTIF or customer fill-rate measurements unless those source fields are provided."))

# -----------------------------
# Planning Agent
# -----------------------------
with tabs[2]:
    st.subheader(f"🧠 {tr('Planning Agent')}")
    st.caption(tr("Turns the decision engine into an ordered sequence of planner actions."))

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Immediate actions", planning_meta["immediate_count"])
    p2.metric("Purchase exposure", f"€{planning_meta['purchase_value']:,.0f}")
    p3.metric("Service risk", f"€{planning_meta['service_risk_value']:,.0f}")
    p4.metric("Risk addressed", f"€{planning_meta['service_risk_addressed']:,.0f}")
    p5.metric("Blocked exposure", f"€{planning_meta['blocked_purchase_exposure']:,.0f}")

    if planning_meta.get("action_changes", 0) > 0:
        st.info(
            f"Change Monitor detects {planning_meta['action_changes']} action changes; "
            f"{planning_meta['change_worsened']} are classified as worsened."
        )

    st.subheader(tr("Recommended execution sequence"))
    st.dataframe(
        planning[[
            "Execution_Priority","SKU","Description","Supplier",
            "Action","Execution_Task","Dependency","Action_Timing",
            "Recommended_Order","Purchase_Value","Service_Risk_Value",
            "Expected_Service_Risk_Addressed","Decision_Confidence"
        ]].head(20),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(tr("Planner rationale"))
    selected_plan_sku = st.selectbox(
        "Explain the recommended action for",
        options=[""] + planning["SKU"].astype(str).tolist()
    )
    if selected_plan_sku:
        r = planning[planning["SKU"].astype(str) == selected_plan_sku].iloc[0]
        st.info(r["Planning_Rationale"])

    st.subheader(tr("Export Planning Agent"))
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "📊 Planning Agent Report (HTML)",
            build_planning_agent_html(planning, planning_meta).encode("utf-8"),
            "planning_agent_report.html",
            "text/html",
            use_container_width=True,
                key="planning_agent_html_tab"
        )
    with e2:
        try:
            planning_xlsx = _excel_planning_agent_bytes(planning, planning_meta)
        except Exception as planning_exc:
            planning_xlsx = None
            st.warning(f"{tr('Excel export unavailable')}: {planning_exc}")
        if planning_xlsx:
            st.download_button(
                "📗 Planning Agent Report (Excel)",
                planning_xlsx,
                "planning_agent_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="planning_agent_excel_tab"
            )

# -----------------------------
# Inventory
# -----------------------------
with tabs[3]:
    st.subheader(tr("Inventory health"))
    left, right = st.columns(2)
    with left:
        status_counts = a["Status"].value_counts()
        st.bar_chart(status_counts)
    with right:
        st.bar_chart(a.groupby("Supplier")["Inventory_Value"].sum())
    inventory_view = a[[
        "SKU","Description","Supplier","Status","Stock","Open_PO",
        "Days_Cover","Safety_Stock","Recommended_Order",
        "Inventory_Value","ABC_XYZ","Action","Action_Timing","Decision_Confidence"
    ]]
    st.dataframe(inventory_view, use_container_width=True, hide_index=True)
    inventory_export = _excel_tab_export_bytes(
        "Inventory", {"Inventory Health": inventory_view},
        {"Inventory value": float(a["Inventory_Value"].sum()), "Service risk": float(a["Service_Risk_Value"].sum()), "Excess exposure": float(a["Excess_Inventory_Value"].sum())}
    )
    if inventory_export:
        st.download_button("📗 Export Inventory to Excel", inventory_export, "inventory_export.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="inventory_excel")

# -----------------------------
# Forecast
# -----------------------------
with tabs[4]:
    st.subheader(tr("Demand outlook"))
    f = a[[
        "SKU","Description","Annual_Sales","Avg_Monthly_Demand",
        "Forecast_Next_Month","Forecast_Change_Pct",
        "Trend_Units_Per_Month","Demand_CV"
    ]].sort_values("Forecast_Next_Month", ascending=False)
    st.dataframe(f, use_container_width=True, hide_index=True)
    forecast_export = _excel_tab_export_bytes(
        "Forecast", {"Demand Outlook": f},
        {"SKUs": int(len(f)), "Next-month forecast": float(f["Forecast_Next_Month"].sum()), "Avg monthly demand": float(f["Avg_Monthly_Demand"].sum())}
    )
    if forecast_export:
        st.download_button("📗 Export Forecast to Excel", forecast_export, "forecast_export.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="forecast_excel")
    st.info(tr("The MVP forecast uses a weighted average of the last 6 months plus a linear trend. The next iteration can add seasonality, intermittent demand and alternative models."))

# -----------------------------
# ABC/XYZ
# -----------------------------
with tabs[5]:
    st.subheader(tr("Segmentation"))
    abc_view = a[[
        "SKU","Description","Annual_Consumption_Value",
        "ABC","Demand_CV","XYZ","ABC_XYZ"
    ]].sort_values("Annual_Consumption_Value", ascending=False)
    st.dataframe(abc_view, use_container_width=True, hide_index=True)
    abc_export = _excel_tab_export_bytes(
        "ABC XYZ", {"ABC XYZ Segmentation": abc_view},
        {"SKUs": int(len(abc_view)), "A class": int((abc_view["ABC"]=="A").sum()), "X class": int((abc_view["XYZ"]=="X").sum())}
    )
    if abc_export:
        st.download_button("📗 Export ABC/XYZ to Excel", abc_export, "abc_xyz_export.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="abc_xyz_excel")

# -----------------------------
# Suppliers
# -----------------------------
with tabs[6]:
    st.subheader(tr("Supplier exposure"))
    sup = a.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU","count"),
        Inventory_Value=("Inventory_Value","sum"),
        Purchase_Value=("Purchase_Value","sum"),
        Critical=("Status", lambda s: (s=="🔴 CRITICAL").sum()),
        Avg_Cover=("Days_Cover","mean")
    ).sort_values(["Critical","Inventory_Value"], ascending=[False,False])
    st.dataframe(sup, use_container_width=True, hide_index=True)
    supplier_export = _excel_tab_export_bytes(
        "Suppliers", {"Supplier Exposure": sup},
        {"Suppliers": int(len(sup)), "Critical SKUs": int(sup["Critical"].sum()), "Purchase exposure": float(sup["Purchase_Value"].sum()), "Inventory value": float(sup["Inventory_Value"].sum())}
    )
    if supplier_export:
        st.download_button("📗 Export Suppliers to Excel", supplier_export, "suppliers_export.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="suppliers_excel")

# -----------------------------
# Scenarios
# -----------------------------
def _scenario_config(demand, volatility, stock, po, cost, moq, safety, service, lead, lead_buffer):
    return {
        "demand": float(demand), "volatility": float(volatility), "stock": float(stock),
        "po": float(po), "cost": float(cost), "moq": float(moq), "safety": int(safety),
        "service": float(service), "lead": float(lead), "lead_buffer": int(lead_buffer)
    }

def _run_saved_scenario(raw_df, cfg):
    sim = raw_df.copy()
    numeric_cols = ["Sales", "Stock", "Open_PO", "Lead_Time_Days", "MOQ", "Unit_Cost"]
    for col in numeric_cols:
        sim[col] = pd.to_numeric(sim[col], errors="coerce").fillna(0)
    sim["Sales"] = sim["Sales"] * cfg["demand"]
    if cfg["volatility"] != 1.0:
        sku_mean = sim.groupby("SKU")["Sales"].transform("mean")
        sim["Sales"] = (sku_mean + (sim["Sales"] - sku_mean) * cfg["volatility"]).clip(lower=0)
    sim["Stock"] = sim["Stock"] * cfg["stock"]
    sim["Open_PO"] = sim["Open_PO"] * cfg["po"]
    sim["Lead_Time_Days"] = sim["Lead_Time_Days"] * cfg["lead"] + cfg["lead_buffer"]
    sim["MOQ"] = sim["MOQ"] * cfg["moq"]
    sim["Unit_Cost"] = sim["Unit_Cost"] * cfg["cost"]
    return analyze(sim, cfg["safety"], cfg["service"])

def _scenario_metrics(base_df, scenario_df):
    base_actions = base_df.set_index("SKU")["Action"].to_dict()
    changed = int(sum(base_actions.get(sku) != action for sku, action in zip(scenario_df["SKU"], scenario_df["Action"])))
    cover = float(scenario_df["Days_Cover"].replace([np.inf, -np.inf], np.nan).median())
    return {
        "Purchase need": float(scenario_df["Purchase_Value"].sum()),
        "Critical SKUs": int((scenario_df["Status"] == "🔴 CRITICAL").sum()),
        "Service risk": float(scenario_df["Service_Risk_Value"].sum()),
        "Excess inventory": float(scenario_df["Excess_Inventory_Value"].sum()),
        "Required stock value": float((scenario_df["Required_Stock"] * scenario_df["Unit_Cost"]).sum()),
        "Median days cover": cover,
        "Action changes": changed,
    }

with tabs[7]:
    st.subheader(f"🧪 {tr('Policy simulator')}")
    st.caption(tr("Simulate decisions before changing the policy."))

    # Scenario controls are deliberately separated from the base planning policy:
    # the original dataset and current policy remain unchanged.
    st.markdown(f"**{tr('Demand & supply assumptions')}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_demand = st.slider(tr("Demand multiplier"), 0.50, 1.50, 1.00, 0.05, key="sim_demand")
        sim_volatility = st.slider(tr("Demand volatility multiplier"), 0.50, 2.00, 1.00, 0.05, key="sim_volatility")
    with c2:
        sim_stock = st.slider(tr("Available stock multiplier"), 0.50, 1.20, 1.00, 0.05, key="sim_stock")
        sim_po = st.slider(tr("Open PO reliability"), 0.00, 1.20, 1.00, 0.05, key="sim_po")
    with c3:
        sim_cost = st.slider(tr("Unit cost multiplier"), 0.80, 1.20, 1.00, 0.05, key="sim_cost")
        sim_moq = st.slider(tr("MOQ multiplier"), 0.50, 2.00, 1.00, 0.05, key="sim_moq")

    st.markdown(f"**{tr('Planning policy')}**")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        sim_safety = st.slider(tr("Safety stock floor"), 0, 90, safety_days, key="sim_safety")
    with p2:
        sim_service = st.slider(tr("Service level"), 0.90, 0.99, float(service), 0.01, key="sim_service")
    with p3:
        sim_lead = st.slider(tr("Lead time multiplier"), .50, 2.00, 1.00, .05, key="sim_lead")
    with p4:
        sim_lead_buffer = st.slider(tr("Lead time buffer (days)"), 0, 30, 0, 1, key="sim_lead_buffer")

    current_cfg = _scenario_config(sim_demand, sim_volatility, sim_stock, sim_po, sim_cost, sim_moq, sim_safety, sim_service, sim_lead, sim_lead_buffer)

    # Save reusable scenarios for side-by-side what-if comparison.
    st.markdown(f"**{tr('Scenario comparison')}**")
    save_col1, save_col2, save_col3 = st.columns([1.4, 1.4, 1])
    with save_col1:
        scenario_slot = st.selectbox(tr("Scenario slot"), ["Scenario A", "Scenario B", "Scenario C"], key="scenario_slot")
    with save_col2:
        default_name = st.session_state.scenario_library.get(scenario_slot, {}).get("name", scenario_slot)
        scenario_name = st.text_input(tr("Scenario name"), value=default_name, key="scenario_name_input")
    with save_col3:
        if st.button(f"💾 {tr('Save scenario')}", use_container_width=True, key="save_scenario"):
            st.session_state.scenario_library[scenario_slot] = {"name": scenario_name.strip() or scenario_slot, "config": current_cfg.copy()}
            st.success(tr("Scenario saved successfully."))

    saved = st.session_state.scenario_library
    if saved:
        selected_slots = st.multiselect(
            tr("Selected scenarios"), list(saved.keys()), default=list(saved.keys()), key="scenario_compare_selection"
        )
        if selected_slots:
            comparison_rows = [{tr("Scenario"): tr("Current policy"), **{tr(k): v for k, v in _scenario_metrics(a, a).items()}}]
            comparison_frames = {tr("Current policy"): a}
            for slot in selected_slots:
                item = saved[slot]
                scenario_df = _run_saved_scenario(raw, item["config"])
                metrics = _scenario_metrics(a, scenario_df)
                label = item["name"] or slot
                comparison_frames[label] = scenario_df
                row = {tr("Scenario"): label}
                for metric_key, value in metrics.items():
                    row[tr(metric_key)] = value
                comparison_rows.append(row)
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown(f"**{tr('Scenario KPI comparison')}**")
            financial_metrics = ["Purchase need", "Service risk", "Excess inventory", "Required stock value"]
            operational_metrics = ["Critical SKUs", "Median days cover", "Action changes"]
            financial_chart = comparison_df[[tr("Scenario")] + [tr(x) for x in financial_metrics]].copy().set_index(tr("Scenario"))
            operational_chart = comparison_df[[tr("Scenario")] + [tr(x) for x in operational_metrics]].copy().set_index(tr("Scenario"))
            st.caption("€ / financial exposure" if st.session_state.get("language", "English") == "English" else "€ / exposición financiera")
            st.bar_chart(financial_chart)
            st.caption("Operational impact" if st.session_state.get("language", "English") == "English" else "Impacto operativo")
            st.bar_chart(operational_chart)

            # Normalized index: current policy = 100 where possible, making the direction of change easy to read.
            normalized = comparison_df.copy()
            for metric_key in financial_metrics + operational_metrics:
                col = tr(metric_key)
                base_value = float(normalized.loc[0, col]) if len(normalized) and pd.notna(normalized.loc[0, col]) else 0
                if base_value != 0:
                    normalized[col] = normalized[col] / base_value * 100
                else:
                    normalized[col] = 0
            normalized_chart = normalized[[tr("Scenario")] + [tr(x) for x in financial_metrics + operational_metrics]].set_index(tr("Scenario"))
            st.caption("Index vs current policy (100 = current policy)" if st.session_state.get("language", "English") == "English" else "Índice vs política actual (100 = política actual)")
            st.bar_chart(normalized_chart)

            comp_export = _excel_tab_export_bytes(
                "Scenario Comparison", {"Scenario KPIs": comparison_df, "Normalized KPIs": normalized.reset_index()},
                {"Scenarios": int(len(comparison_df))}
            )
            if comp_export:
                st.download_button(
                    "📗 Export Scenario Comparison to Excel", comp_export, "scenario_comparison.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, key="scenario_comparison_excel"
                )
    else:
        st.caption(tr("No saved scenarios yet."))

    # Build a deterministic scenario dataset. Demand volatility changes the
    # dispersion around each SKU's mean, allowing a more realistic risk test.
    sim = raw.copy()
    numeric_cols = ["Sales", "Stock", "Open_PO", "Lead_Time_Days", "MOQ", "Unit_Cost"]
    for col in numeric_cols:
        sim[col] = pd.to_numeric(sim[col], errors="coerce").fillna(0)

    sim["Sales"] = sim["Sales"] * sim_demand
    if sim_volatility != 1.0:
        sku_mean = sim.groupby("SKU")["Sales"].transform("mean")
        sim["Sales"] = (sku_mean + (sim["Sales"] - sku_mean) * sim_volatility).clip(lower=0)
    sim["Stock"] = sim["Stock"] * sim_stock
    sim["Open_PO"] = sim["Open_PO"] * sim_po
    sim["Lead_Time_Days"] = sim["Lead_Time_Days"] * sim_lead + sim_lead_buffer
    sim["MOQ"] = sim["MOQ"] * sim_moq
    sim["Unit_Cost"] = sim["Unit_Cost"] * sim_cost

    sim_a = analyze(sim, sim_safety, sim_service)
    base_val = float(a["Purchase_Value"].sum())
    sim_val = float(sim_a["Purchase_Value"].sum())
    base_risk = int((a["Status"] == "🔴 CRITICAL").sum())
    sim_risk = int((sim_a["Status"] == "🔴 CRITICAL").sum())
    base_service_risk = float(a["Service_Risk_Value"].sum())
    sim_service_risk = float(sim_a["Service_Risk_Value"].sum())
    base_excess = float(a["Excess_Inventory_Value"].sum())
    sim_excess = float(sim_a["Excess_Inventory_Value"].sum())
    base_required = float((a["Required_Stock"] * a["Unit_Cost"]).sum())
    sim_required = float((sim_a["Required_Stock"] * sim_a["Unit_Cost"]).sum())
    base_cover = float(a["Days_Cover"].replace([np.inf, -np.inf], np.nan).median())
    sim_cover = float(sim_a["Days_Cover"].replace([np.inf, -np.inf], np.nan).median())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr("Purchase need"), f"€{sim_val:,.0f}", f"€{sim_val-base_val:+,.0f}")
    m2.metric(tr("Critical SKUs"), sim_risk, f"{sim_risk-base_risk:+d}")
    m3.metric(tr("Service risk"), f"€{sim_service_risk:,.0f}", f"€{sim_service_risk-base_service_risk:+,.0f}")
    m4.metric(tr("Required stock value"), f"€{sim_required:,.0f}", f"€{sim_required-base_required:+,.0f}")

    st.markdown(f"**{tr('Scenario impact')}**")
    impact = pd.DataFrame([
        [tr("Purchase need"), base_val, sim_val, sim_val-base_val],
        [tr("Critical SKUs"), base_risk, sim_risk, sim_risk-base_risk],
        [tr("Service risk"), base_service_risk, sim_service_risk, sim_service_risk-base_service_risk],
        [tr("Excess inventory"), base_excess, sim_excess, sim_excess-base_excess],
        [tr("Required stock value"), base_required, sim_required, sim_required-base_required],
        [tr("Median days cover"), base_cover, sim_cover, sim_cover-base_cover],
    ], columns=[tr("Metric"), tr("Current policy"), tr("Scenario"), tr("Change")])
    st.dataframe(impact, use_container_width=True, hide_index=True)

    st.markdown(f"**{tr('Scenario decision impact')}**")
    scenario_view = sim_a[[
        "SKU","Description","Supplier","Status","Action","Days_Cover",
        "Lead_Time_Days","Recommended_Order","Purchase_Value",
        "Service_Risk_Value","Excess_Inventory_Value","Decision_Confidence"
    ]].copy().sort_values(["Status","Purchase_Value"], ascending=[True, False])
    st.dataframe(scenario_view.head(100), use_container_width=True, hide_index=True)

    base_actions = a.set_index("SKU")["Action"].to_dict()
    changed_actions = int(sum(base_actions.get(sku) != action for sku, action in zip(sim_a["SKU"], sim_a["Action"])))
    c1, c2, c3 = st.columns(3)
    c1.metric(tr("Action changes"), changed_actions)
    c2.metric(tr("Excess inventory"), f"€{sim_excess:,.0f}", f"€{sim_excess-base_excess:+,.0f}")
    c3.metric(tr("Median days cover"), f"{sim_cover:.1f} d", f"{sim_cover-base_cover:+.1f} d")

    st.info(tr("Scenario results are simulations only. Validate the impact before changing the live planning policy."))


def _excel_data_quality_bytes(raw, dq):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    title = wb.add_format({"bold": True, "font_size": 18, "font_color": "#FFFFFF", "bg_color": "#17365D", "align": "left", "valign": "vcenter"})
    subtitle = wb.add_format({"italic": True, "font_color": "#666666"})
    header = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D", "align": "center", "valign": "vcenter", "text_wrap": True})
    ok_fmt = wb.add_format({"font_color": "#166534", "bg_color": "#DCFCE7"})
    warn_fmt = wb.add_format({"font_color": "#92400E", "bg_color": "#FEF3C7"})
    crit_fmt = wb.add_format({"font_color": "#991B1B", "bg_color": "#FEE2E2"})
    text_fmt = wb.add_format({"valign": "top", "text_wrap": True})
    integer_fmt = wb.add_format({"num_format": "#,##0", "valign": "top"})

    summary = data_quality_summary(raw, dq)
    ws = wb.add_worksheet(tr("Summary"))
    ws.hide_gridlines(2)
    ws.merge_range("A1:F1", "Supply Chain AI — Data Quality Report", title)
    ws.write("A2", f"Latest period: {summary['latest_period']} · {summary['rows']:,} rows · {summary['skus']:,} SKUs · {summary['suppliers']:,} suppliers", subtitle)
    kpis = [("Rows", summary["rows"]), ("SKUs", summary["skus"]), ("Suppliers", summary["suppliers"]), ("Checks", summary["checks"]), ("Warnings", summary["warnings"]), ("Critical", summary["critical"])]
    for i, (label, value) in enumerate(kpis):
        col = i % 3 * 2
        row = 3 + (i // 3) * 2
        ws.write(row, col, label, header)
        ws.write(row + 1, col, value, integer_fmt)
        ws.set_column(col, col, 18)
        ws.set_column(col + 1, col + 1, 3)
    status_text = "ALL CHECKS PASSED" if summary["critical"] == 0 and summary["warnings"] == 0 else ("CRITICAL ISSUES DETECTED" if summary["critical"] > 0 else "WARNINGS DETECTED")
    status_fmt = crit_fmt if summary["critical"] > 0 else (warn_fmt if summary["warnings"] > 0 else ok_fmt)
    ws.write(8, 0, status_text, status_fmt)
    ws.merge_range(8, 0, 8, 5, status_text, status_fmt)
    ws.set_row(8, 24)

    detail = wb.add_worksheet(tr("Quality Checks"))
    detail.hide_gridlines(2)
    detail.write_row(0, 0, ["Category", "Check", "Status", "Count", "Details"], header)
    for r, row in enumerate(dq[["Category","Check","Status","Count","Details"]].itertuples(index=False, name=None), 1):
        detail.write(r, 0, row[0], text_fmt)
        detail.write(r, 1, row[1], text_fmt)
        fmt = crit_fmt if row[2] == "CRITICAL" else (warn_fmt if row[2] == "WARNING" else ok_fmt)
        detail.write(r, 2, row[2], fmt)
        detail.write(r, 3, 0 if pd.isna(row[3]) else row[3], integer_fmt)
        detail.write(r, 4, row[4], text_fmt)
    detail.add_table(0, 0, len(dq), 4, {"name": "DataQualityChecks", "style": "Table Style Medium 2", "columns": [{"header": c} for c in ["Category","Check","Status","Count","Details"]]})
    detail.set_column("A:A", 18); detail.set_column("B:B", 32); detail.set_column("C:C", 14); detail.set_column("D:D", 12); detail.set_column("E:E", 60)
    detail.freeze_panes(1, 0)
    wb.close()
    buf.seek(0)
    return buf.getvalue()



# -----------------------------
# MRP / Manufacturing
# -----------------------------
with tabs[8]:
    st.subheader(f"🏭 {tr('Material Requirements Planning')}")
    st.caption(tr("Convert demand into a manufacturing and component plan."))
    st.info(tr("The MRP engine uses the selected finished-goods forecast, BOM quantities, current component inventory and open POs to calculate time-phased material needs."))

    up1, up2 = st.columns([2,1])
    with up1:
        bom_upload = st.file_uploader(
            tr("Upload BOM"), type=["csv","xlsx","xls"],
            key="mrp_bom_uploader", help=tr("BOM fields: Parent_SKU, Component_SKU, Qty_Per, Scrap_Pct")
        )
    with up2:
        template_bytes, template_type = _mrp_template_bytes()
        st.download_button(
            f"📥 {tr('Download BOM template')}", template_bytes,
            ("bom_template.xlsx" if template_type=="xlsx" else "bom_template.csv") if st.session_state.language=="English" else ("plantilla_bom.xlsx" if template_type=="xlsx" else "plantilla_bom.csv"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if template_type=="xlsx" else "text/csv",
            use_container_width=True, key="mrp_bom_template"
        )

    if bom_upload is not None:
        try:
            uploaded_bom = normalize_bom(read_uploaded(bom_upload))
            if uploaded_bom.empty:
                st.error(tr("BOM file must contain Parent_SKU, Component_SKU and Qty_Per columns."))
            else:
                st.session_state.mrp_bom = uploaded_bom
                st.session_state.mrp_result = None
                st.success(tr("BOM uploaded successfully."))
        except Exception as exc:
            st.error(f"{tr('BOM upload failed')}: {exc}")

    bom = st.session_state.mrp_bom
    if bom is None or bom.empty:
        st.warning(tr("No BOM loaded. Use the template to prepare your manufacturing structure."))
    else:
        b1,b2,b3,b4=st.columns(4)
        b1.metric(tr("BOM lines"), len(bom))
        b2.metric(tr("Parent SKUs"), bom["Parent_SKU"].nunique())
        b3.metric(tr("Components"), bom["Component_SKU"].nunique())
        b4.metric(tr("Max scrap %"), f"{bom['Scrap_Pct'].max():.1f}%")
        st.dataframe(bom[["Parent_SKU","Component_SKU","Qty_Per","Scrap_Pct"]].head(100), use_container_width=True, hide_index=True)
        dataset_skus=set(a_source["SKU"].astype(str))
        missing_components=sorted(set(bom["Component_SKU"].astype(str))-dataset_skus)
        missing_parents=sorted(set(bom["Parent_SKU"].astype(str))-dataset_skus)
        if missing_components:
            st.warning(f"⚠️ {tr('BOM components not found in the dataset')}: {len(missing_components)}")
        if missing_parents:
            st.warning(f"⚠️ {tr('BOM parents not found in the dataset')}: {len(missing_parents)}")

        st.markdown(f"### ⚙️ {tr('MRP planning parameters')}")
        p1,p2,p3,p4=st.columns(4)
        with p1: horizon=st.slider(tr("Planning horizon (months)"),3,12,6,key="mrp_horizon")
        with p2: demand_growth=st.slider(tr("Monthly demand growth"),-0.20,0.20,0.0,0.01,key="mrp_growth",format="%+.0f%%")
        with p3: prod_safety=st.slider(tr("Production safety stock (days)"),0,30,5,key="mrp_safety")
        with p4: use_open_po=st.checkbox(tr("Use open POs as month-1 receipts"),True,key="mrp_open_po")

        parent_options=sorted(bom["Parent_SKU"].unique().tolist())
        available_parent=[x for x in parent_options if x in set(a_source["SKU"].astype(str))]
        selected_parents=st.multiselect(tr("Parent SKUs"),available_parent,default=available_parent[:min(50,len(available_parent))],key="mrp_parents",placeholder=tr("All parent SKUs"))
        if not selected_parents: selected_parents=available_parent

        if st.button(f"▶️ {tr('Run MRP')}", type="primary", use_container_width=True, key="run_mrp"):
            with st.spinner(tr("Run MRP")):
                full_a=analyze(raw,safety_days,service)
                mps_df,mrp_df,mrp_meta=run_mrp(raw,full_a,bom,selected_parents,horizon,demand_growth,prod_safety,use_open_po)
            st.session_state.mrp_result=(mps_df,mrp_df,mrp_meta)

        result=st.session_state.get("mrp_result")
        if not result:
            st.caption(tr("Run MRP"))
        else:
            mps_df,mrp_df,mrp_meta=result
            st.markdown(f"### 📊 {tr('Manufacturing KPIs')}")
            k1,k2,k3,k4,k5=st.columns(5)
            k1.metric(tr("Manufacturing orders"), f"{int((mps_df['Planned_Production']>0).sum()):,}")
            k2.metric(tr("Components"), f"{mrp_meta['components']:,}")
            k3.metric(tr("Components with shortage"), f"{mrp_meta['shortages']:,}")
            k4.metric(tr("Total component requirement"), f"{mrp_df['Gross_Requirement'].sum():,.0f}")
            k5.metric(tr("Total planned purchase"), f"€{mrp_meta['purchase']:,.0f}")

            if not mrp_df.empty:
                st.markdown(f"### 🏭 {tr('Planned production')}")
                prod_view=mps_df.rename(columns={"Parent_SKU":tr("Parent SKU"),"Period":tr("Period"),"Gross_Requirement":tr("Gross requirements"),"Planned_Production":tr("Planned production")})
                st.dataframe(prod_view,use_container_width=True,hide_index=True)

                st.markdown(f"### 🧩 {tr('Component requirements')}")
                comp_summary=mrp_df.groupby("Component_SKU",as_index=False).agg(
                    Gross_Requirement=("Gross_Requirement","sum"),
                    Net_Requirement=("Net_Requirement","sum"),
                    Planned_Order_Receipt=("Planned_Order_Receipt","sum"),
                    Shortage=("Shortage","sum"),
                    Planned_Purchase_Value=("Planned_Purchase_Value","sum"),
                    Supplier=("Supplier","first"), Lead_Time_Days=("Lead_Time_Days","first"), MOQ=("MOQ","first")
                ).sort_values(["Shortage","Planned_Purchase_Value"],ascending=[False,False])
                st.dataframe(comp_summary.rename(columns={"Component_SKU":tr("Component SKU"),"Gross_Requirement":tr("Gross requirements"),"Net_Requirement":tr("Net requirements"),"Planned_Order_Receipt":tr("Planned order receipts"),"Shortage":tr("Shortage"),"Planned_Purchase_Value":tr("Total planned purchase"),"Supplier":tr("Supplier"),"Lead_Time_Days":tr("Lead time"),"MOQ":tr("MOQ")}),use_container_width=True,hide_index=True)

                st.markdown(f"### ⚠️ {tr('MRP exception messages')}")
                exceptions=mrp_df[mrp_df["Shortage"]>0].copy()
                if exceptions.empty:
                    st.success(tr("No material shortages detected in the simulated horizon."))
                else:
                    st.warning(tr("Material shortages requiring action."))
                    ex=exceptions[["Component_SKU","Period","Shortage","Supplier","Lead_Time_Days"]].head(100).copy()
                    ex.columns=[tr("Component SKU"),tr("Period"),tr("Shortage"),tr("Supplier"),tr("Lead time")]
                    st.dataframe(ex,use_container_width=True,hide_index=True)

                st.markdown(f"### 📅 {tr('MRP results')}")
                detail=mrp_df.copy()
                detail.columns=[tr("Component SKU"),tr("Period"),tr("Opening inventory"),tr("Gross requirements"),tr("Scheduled receipts"),tr("Net requirements"),tr("Planned order receipts"),tr("Projected ending inventory"),tr("Shortage"),tr("Supplier"),tr("Lead time"),tr("MOQ"),tr("Unit cost"),tr("Total planned purchase"),tr("Release Period"),tr("Action")]
                st.dataframe(detail,use_container_width=True,hide_index=True)

                mrp_export=_mrp_export_bytes(mps_df,mrp_df,mrp_meta,st.session_state.language)
                if mrp_export:
                    st.download_button(f"📗 {tr('Export MRP')}",mrp_export,"mrp_manufacturing_plan.xlsx" if st.session_state.language=="English" else "plan_mrp_fabricacion.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True,key="mrp_export")
            st.info(tr("MRP is a planning simulation. Validate BOMs, routings, calendars and supplier due dates before execution."))
# -----------------------------
# Data Quality
# -----------------------------
with tabs[9]:
    st.subheader(f"🧹 {tr('Data Quality')}")
    st.caption(tr("Checks the data before operational decisions are used."))

    dq_summary = data_quality_summary(raw, dq)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", dq_summary["rows"])
    q2.metric("SKUs", dq_summary["skus"])
    q3.metric("Suppliers", dq_summary["suppliers"])
    q4.metric("Warnings", dq_summary["warnings"])
    q5.metric("Critical", dq_summary["critical"])

    if dq_summary["critical"] > 0:
        st.error(tr("⛔ Critical data-quality issues detected."))
    elif dq_summary["warnings"] > 0:
        st.warning(tr("⚠️ Data-quality warnings detected. Review them before issuing purchase decisions."))
    else:
        st.success(tr("✅ All current data-quality checks passed."))

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(tr("Latest period"), dq_summary["latest_period"])
        st.metric(tr("Checks completed"), dq_summary["checks"])
    with c2:
        st.dataframe(
            dq[["Category","Check","Status","Count","Details"]],
            use_container_width=True, hide_index=True
        )

    st.subheader(f"📤 {tr('Export Data Quality')}")
    dq_html = build_data_quality_report_html(raw, dq)
    dq1, dq2 = st.columns(2)
    with dq1:
        st.download_button(
            "🌐 Data Quality Report (HTML)",
            dq_html.encode("utf-8"),
            "data_quality_report.html",
            "text/html",
            use_container_width=True,
            key="data_quality_html_export"
        )
    with dq2:
        try:
            dq_excel = _excel_data_quality_bytes(raw, dq)
        except Exception as dq_exc:
            dq_excel = None
            st.warning(f"{tr('Excel export unavailable')}: {dq_exc}")
        if dq_excel:
            st.download_button(
                "📗 Data Quality Report (Excel)",
                dq_excel,
                "data_quality_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="data_quality_excel_export"
            )
    st.caption(tr("HTML provides the management-ready visual report; Excel provides editable quality checks and KPI summary."))


# -----------------------------
# Action Plan
# -----------------------------
with tabs[10]:
    st.subheader(f"📝 {tr('Weekly Action Plan')}")
    st.caption(tr("Planner-ready worklist generated by the Decision Engine."))

    f1, f2, f3 = st.columns(3)
    with f1:
        action_filter = st.multiselect(
            "Action",
            ["BUY_NOW","CONFIRM_PO","REVIEW","DO_NOT_BUY","MONITOR"],
            default=["BUY_NOW","CONFIRM_PO","REVIEW","DO_NOT_BUY"]
        )
    with f2:
        owner_filter = st.multiselect(
            "Owner",
            sorted(plan["Owner"].unique().tolist()),
            default=sorted(plan["Owner"].unique().tolist())
        )
    with f3:
        deadline_filter = st.multiselect(
            "Deadline",
            sorted(plan["Deadline"].unique().tolist()),
            default=sorted(plan["Deadline"].unique().tolist())
        )

    plan_view = plan.copy()
    action_map = a.set_index("SKU")["Action"].to_dict()
    plan_view["Action_Code"] = plan_view["SKU"].map(action_map)
    plan_view = plan_view[
        plan_view["Action_Code"].isin(action_filter)
        & plan_view["Owner"].isin(owner_filter)
        & plan_view["Deadline"].isin(deadline_filter)
    ]

    m1, m2, m3 = st.columns(3)
    m1.metric("Actions", len(plan_view))
    m2.metric("Immediate", int(plan_view["Deadline"].eq("Today").sum()))
    m3.metric("Purchase value", f"€{plan_view['Purchase_Value'].sum():,.0f}")

    st.dataframe(
        plan_view.drop(columns=["Action_Code"]),
        use_container_width=True, hide_index=True
    )

    st.subheader(tr("Supplier follow-up"))
    supplier_skus = plan_view[plan_view["Action_Code"].isin(["BUY_NOW","CONFIRM_PO"])]["SKU"].astype(str).tolist()
    selected_sku = st.selectbox(
        "Generate supplier communication",
        options=[""] + supplier_skus
    )
    if selected_sku:
        row = a[a["SKU"].astype(str) == selected_sku].iloc[0]
        message = supplier_message(row)
        st.code(message, language="text")
        st.download_button(
            "⬇️ Download supplier message",
            message.encode("utf-8"),
            f"supplier_message_{selected_sku}.txt",
            "text/plain"
        )

    action_export_df = plan_view.drop(columns=["Action_Code"])
    st.markdown(f"### 📤 {tr('Export Action Plan')}")
    ax1, ax2 = st.columns(2)
    with ax1:
        action_html = build_action_report_html(action_export_df)
        st.download_button(
            "🌐 Download Action Plan Report (HTML)",
            action_html.encode("utf-8"),
            "weekly_action_plan_report.html",
            "text/html",
            use_container_width=True, key="action_plan_html"
        )
    with ax2:
        action_xlsx = _excel_tab_export_bytes(
            "Weekly Action Plan",
            {
                "Action Plan": action_export_df,
                "Supplier Summary": action_export_df.groupby("Supplier", as_index=False).agg(
                    Actions=("SKU","count"), Purchase_Value=("Purchase_Value","sum")
                ).sort_values("Purchase_Value", ascending=False)
            },
            {
                "Actions": int(len(action_export_df)),
                "Immediate": int(action_export_df["Deadline"].eq("Today").sum()),
                "Purchase value": float(action_export_df["Purchase_Value"].sum()),
                "Owners": int(action_export_df["Owner"].nunique()),
            }
        )
        if action_xlsx:
            st.download_button(
                "📗 Download Action Plan Report (Excel)",
                action_xlsx,
                "weekly_action_plan_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="action_plan_excel"
            )
    st.caption(tr("HTML and Excel use the current filtered view. HTML includes KPIs and an executive presentation; Excel includes Summary, Action Plan and Supplier Summary with filters."))

# -----------------------------
# Change Monitor
# -----------------------------
with tabs[11]:
    st.subheader(tr("🔄 What changed?"))
    if not comparison_meta.get("has_comparison"):
        st.info(tr("Need at least two historical periods to compare evolution."))
    else:
        st.caption(f"{tr('Comparing')} **{comparison_meta['previous_period']} → {comparison_meta['current_period']}**")

        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("Purchase need", f"€{comparison_meta['purchase_current']:,.0f}", f"{comparison_meta['purchase_delta']:+,.0f} €")
        cm2.metric("Service risk", f"€{comparison_meta['service_risk_current']:,.0f}", f"{comparison_meta['service_risk_delta']:+,.0f} €")
        cm3.metric("Excess exposure", f"€{comparison_meta['excess_current']:,.0f}", f"{comparison_meta['excess_delta']:+,.0f} €")
        cm4.metric("Critical SKUs", comparison_meta["critical_current"], f"{comparison_meta['critical_delta']:+d}")
        cm5.metric("Action changes", comparison_meta["action_changes"], f"{comparison_meta['worsened']} worsened")

        if comparison_meta["worsened"] > comparison_meta["improved"]:
            st.warning(f"{tr('There are more unfavorable changes')} ({comparison_meta['worsened']}) {tr('than favorable changes')} ({comparison_meta['improved']}).")
        elif comparison_meta["improved"] > comparison_meta["worsened"]:
            st.success(f"{tr('The evolution is mostly favorable')}: {comparison_meta['improved']} {tr('improved')} {tr('versus')} {comparison_meta['worsened']} {tr('worsened')}.")
        else:
            st.info(tr("The evolution is balanced between improvements and deteriorations."))

        view = comparison[[
            "Priority","SKU","Description","Supplier","Previous_Action","Current_Action",
            "Action_Transition","Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
            "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta","Sales_Delta_Pct",
            "Change_Classification","Change_Reason"
        ]].copy()
        st.dataframe(view, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(tr("Top worsened / watch"))
            worsened = comparison[comparison["Change_Classification"].isin(["WORSENED","WATCH"])].head(8)
            st.dataframe(
                worsened[["SKU","Description","Current_Action","Days_Cover_Delta",
                          "Purchase_Value_Delta","Service_Risk_Delta","Change_Classification"]],
                use_container_width=True, hide_index=True
            )
        with c2:
            st.subheader(tr("Top improved"))
            improved = comparison[comparison["Change_Classification"]=="IMPROVED"].sort_values(
                ["Service_Risk_Delta","Purchase_Value_Delta"], ascending=[True,True]
            ).head(8)
            st.dataframe(
                improved[["SKU","Description","Previous_Action","Current_Action",
                          "Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta"]],
                use_container_width=True, hide_index=True
            )

        change_report_html = build_change_monitor_html(comparison, comparison_meta)

        st.markdown(tr("### 📤 Export Change Monitor"))
        export_cm1, export_cm2 = st.columns(2)

        with export_cm1:
            st.download_button(
                "📊 Download Change Monitor Report (HTML)",
                change_report_html.encode("utf-8"),
                "change_monitor_report.html",
                "text/html",
                use_container_width=True
            )

        try:
            change_report_xlsx = _excel_change_monitor_bytes(comparison, comparison_meta)
        except Exception as export_exc:
            change_report_xlsx = None
            st.warning(f"{tr('Excel export unavailable')}: {export_exc}")

        with export_cm2:
            if change_report_xlsx:
                st.download_button(
                    "📗 Download Change Monitor Report (Excel)",
                    change_report_xlsx,
                    "change_monitor_report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        st.caption(
            "HTML and Excel contain the same Change Monitor analysis: KPIs, period deltas, "
            "top changes, classification and reasons."
        )


# -----------------------------
# Copilot
# -----------------------------
with tabs[12]:
    st.subheader("🤖 Your Supply Chain Copilot")
    if st.session_state.get("copilot_prefill"):
        pending = st.session_state["copilot_prefill"]
        st.info(f"**{tr('Prepared analysis')}:** {pending}")
        pc1, pc2 = st.columns(2)
        if pc1.button("▶️ Run prepared analysis", use_container_width=True, key="run_prefill"):
            st.session_state.chat.append({"role":"user", "content":pending})
            st.session_state.copilot_prefill = ""
            with st.chat_message("user"):
                st.markdown(pending)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    ans = ai_chat(pending, a, effective_key, model, comparison, comparison_meta, raw)
                st.markdown(ans)
            st.session_state.chat.append({"role":"assistant", "content":ans})
        if pc2.button("✖️ Clear prepared analysis", use_container_width=True, key="clear_prefill"):
            st.session_state.copilot_prefill = ""
            st.rerun()

    st.markdown("### ⚡ Quick analyses")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    quick_question = None
    if q1.button("🛒 Purchase priorities"):
        quick_question = tr("What should I buy this week and what are the 3 most important priorities?")
    if q2.button("💰 Reduce inventory"):
        quick_question = tr("Where can I reduce inventory without materially increasing service risk?")
    if q3.button("🚚 Supplier risk"):
        quick_question = tr("Which suppliers require the most attention and why?")
    if q4.button("📈 Demand outlook"):
        quick_question = tr("What demand changes could alter my purchase decisions?")
    if q5.button("🔄 What changed?"):
        quick_question = tr("What changed between the latest period and the previous one, and what are the 3 largest variations?")
    if q6.button("📊 Logistics KPIs"):
        quick_question = tr("What is the state of the main logistics KPIs and which ones require attention?")
    if quick_question:
        st.session_state.chat.append({"role": "user", "content": quick_question})
        with st.chat_message("user"):
            st.markdown(quick_question)
        with st.chat_message("assistant"):
            with st.spinner(tr("Running Supply Chain analysis...")):
                ans = ai_chat(quick_question, a, effective_key, model, comparison, comparison_meta, raw)
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})

    st.subheader("🤖 Ask your Supply Chain Copilot")
    st.caption("Examples: “What should I buy this week?”, “Where is my biggest stockout risk?”, “Which suppliers need attention?”")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask a supply-chain question…")
    if q:
        st.session_state.chat.append({"role":"user","content":q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                ans = ai_chat(q, a, effective_key, model, comparison, comparison_meta, raw)
            st.markdown(ans)
        st.session_state.chat.append({"role":"assistant","content":ans})

# -----------------------------
# Export
# -----------------------------
with tabs[13]:
    st.subheader("📤 Reporting Center")
    st.caption(tr("Visual HTML reports and professional Excel workbooks containing the same decision-ready information."))

    pack_bytes, report_map = build_management_pack(a, raw, dq, plan, comparison, comparison_meta)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Purchase requirement", f"€{a['Purchase_Value'].sum():,.0f}")
    r2.metric("Service risk", f"€{a['Service_Risk_Value'].sum():,.0f}")
    r3.metric("Excess exposure", f"€{a['Excess_Inventory_Value'].sum():,.0f}")
    r4.metric("Actions", len(plan))

    excel_exec = excel_detail = excel_complete = None
    excel_error = None
    try:
        excel_exec = _excel_export_bytes("executive", a, raw, dq, plan, comparison, comparison_meta)
        excel_detail = _excel_export_bytes("detailed", a, raw, dq, plan, comparison, comparison_meta)
        excel_complete = _excel_export_bytes("complete", a, raw, dq, plan, comparison, comparison_meta)
    except Exception as e:
        excel_error = str(e)

    st.markdown("### 1. Executive Report")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📊 Executive Report (HTML)",
            report_map["01_Executive_Report.html"].encode("utf-8"),
            "supply_chain_executive_report.html",
            "text/html", use_container_width=True
        )
    with c2:
        if excel_exec:
            st.download_button(
                "📗 Executive Report (Excel)",
                excel_exec,
                "supply_chain_executive_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    st.caption("Same executive KPIs and priorities, with editable tables, formatting and charts in Excel.")

    st.markdown(tr("### 2. Detailed visual reports"))
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📊 Inventory & Service Risk (HTML)", report_map["02_Inventory_Risk_Report.html"].encode("utf-8"), "inventory_service_risk_report.html", "text/html", use_container_width=True)
        st.download_button("🛒 Purchase Plan (HTML)", report_map["03_Purchase_Plan_Report.html"].encode("utf-8"), "purchase_plan_report.html", "text/html", use_container_width=True)
        st.download_button("🚚 Supplier Risk (HTML)", report_map["05_Supplier_Risk_Report.html"].encode("utf-8"), "supplier_risk_report.html", "text/html", use_container_width=True)
    with d2:
        st.download_button("📝 Weekly Action Plan (HTML)", report_map["04_Action_Plan_Report.html"].encode("utf-8"), "weekly_action_plan_report.html", "text/html", use_container_width=True)
        st.download_button("🧹 Data Quality (HTML)", report_map["06_Data_Quality_Report.html"].encode("utf-8"), "data_quality_report.html", "text/html", use_container_width=True)
        if excel_detail:
            st.download_button(
                "📗 Detailed Reports (Excel)",
                excel_detail,
                "supply_chain_detailed_reports.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    st.caption("The Excel workbook mirrors the detailed reports and adds a Change Monitor sheet.")

    if "07_Change_Monitor_Report.html" in report_map:
        st.markdown("### 3. Change Monitor")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.download_button(
                "🔄 Change Monitor Report (HTML)",
                report_map["07_Change_Monitor_Report.html"].encode("utf-8"),
                "change_monitor_report.html",
                "text/html", use_container_width=True
            )
        with cm2:
            st.metric(
                "Changes",
                comparison_meta["action_changes"],
                f"{comparison_meta['worsened']} worsened / {comparison_meta['improved']} improved"
            )

    st.markdown("### 4. Planning Agent")
    pe1, pe2 = st.columns(2)
    with pe1:
        st.download_button(
            "📊 Planning Agent Report (HTML)",
            build_planning_agent_html(planning, planning_meta).encode("utf-8"),
            "planning_agent_report.html",
            "text/html",
            use_container_width=True,
                key="planning_agent_html_export"
        )
    with pe2:
        try:
            planning_export_xlsx = _excel_planning_agent_bytes(planning, planning_meta)
        except Exception as planning_export_exc:
            planning_export_xlsx = None
            st.warning(f"{tr('Excel export unavailable')}: {planning_export_exc}")
        if planning_export_xlsx:
            st.download_button(
                "📗 Planning Agent Report (Excel)",
                planning_export_xlsx,
                "planning_agent_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="planning_agent_excel_export"
            )

    st.markdown("### 5. Complete Management Pack")
    p1, p2 = st.columns(2)
    with p1:
        st.download_button("📦 Management Pack (ZIP)", pack_bytes, "supply_chain_management_pack_v17.zip", "application/zip", use_container_width=True)
    with p2:
        if excel_complete:
            st.download_button(
                "📗 Complete Management Pack (Excel)",
                excel_complete,
                "supply_chain_complete_management_pack.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    st.caption("The Excel pack combines the executive dashboard, detailed report sheets, Change Monitor and normalized source data in one workbook.")

    if excel_error:
        st.warning(f"{tr('Excel export unavailable')}: {excel_error}")

    st.info("Raw CSV exports remain removed from the reporting workflow. HTML and Excel are now the primary shareable outputs.")


st.divider()
st.caption(tr("Supply Chain AI Copilot V2.0.13 — recommendations require planner validation before execution."))
