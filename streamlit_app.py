
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
    return (
        f"Subject: Urgent supply chain follow-up — {row['SKU']} / {row['Description']}\n\n"
        f"Hello {row['Supplier']} team,\n\n"
        f"We are reviewing replenishment for {row['SKU']} ({row['Description']}). "
        f"The current stock coverage is {row['Days_Cover']:.1f} days and the lead time is "
        f"{row['Lead_Time_Days']:.0f} days.\n\n"
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

def _html_shell(title, subtitle, body):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
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
<div class="meta">Generated {generated} · Supply Chain AI Copilot V2.0.6</div>
</div>
{body}
<div class="footer">Decision support only. Validate purchase execution and supplier commitments before release.</div>
</div>
</body>
</html>"""

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
            '<div class="section"><div class="note">No hay dos periodos comparables disponibles.</div></div>'
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

    for j, col in enumerate(columns):
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
            "columns": [{"header": c} for c in columns],
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

    ws = wb.add_worksheet("Executive")
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

    action_ws = wb.add_worksheet("Action Plan")
    _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
    _xlsx_write_df(
        action_ws, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "ExecutiveActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    dq_ws = wb.add_worksheet("Data Quality")
    _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecutiveDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_detailed(a, raw, dq, plan):
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)

    inv = wb.add_worksheet("Inventory Risk")
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

    pur = wb.add_worksheet("Purchase Plan")
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

    act = wb.add_worksheet("Action Plan")
    _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
    _xlsx_write_df(
        act, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "DetailedActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    sup = wb.add_worksheet("Supplier Risk")
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

    dq_ws = wb.add_worksheet("Data Quality")
    _xlsx_title(dq_ws, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_complete(a, raw, dq, plan):
    wb = _xlsx_build_detailed(a, raw, dq, plan)

    # Complete pack adds Executive + Source Data to the detailed workbook.
    ex = wb.add_worksheet("Executive")
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

    src_ws = wb.add_worksheet("Source Data")
    _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
    _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "SourceData")

    return wb



def _excel_planning_agent_bytes(planning, meta):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)
    ws = wb.add_worksheet("Planning Agent")

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
    ws = wb.add_worksheet("Change Monitor")

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
        ws = wb.add_worksheet("Executive")
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
        action_ws = wb.add_worksheet("Action Plan")
        _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
        _xlsx_write_df(action_ws, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "ExecActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        dq_ws = wb.add_worksheet("Data Quality")
        _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
        _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecDQ", widths={"Check":30,"Details":42})

    elif kind == "detailed":
        fmt = _xlsx_base_formats(wb)
        inv = wb.add_worksheet("Inventory Risk")
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "InventoryRisk", formats={**fmt, "Excess_Inventory_Value": fmt["currency"], "Service_Risk_Value": fmt["currency"]}, widths={"Description":30})
        _xlsx_write_section_chart(inv, wb, "bar", "Excess inventory value", 0, 9, 4, min(3+len(x),13), "M4")
        pur = wb.add_worksheet("Purchase Plan")
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "PurchasePlan", formats={**fmt, "Unit_Cost": fmt["currency"], "Purchase_Value": fmt["currency"], "Recommended_Order":fmt["integer"], "Open_PO":fmt["integer"]}, widths={"Description":30})
        _xlsx_write_section_chart(pur, wb, "column", "Purchase value by SKU", 0, 6, 4, min(3+len(x),13), "M4")
        act = wb.add_worksheet("Action Plan")
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "DetailedActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        sup = wb.add_worksheet("Supplier Risk")
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
        dqs = wb.add_worksheet("Data Quality")
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDQ", widths={"Check":30,"Details":42})

    else:
        fmt = _xlsx_base_formats(wb)
        ex = wb.add_worksheet("Executive")
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
        inv = wb.add_worksheet("Inventory Risk")
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "CompleteInventoryRisk", formats={**fmt,"Excess_Inventory_Value":fmt["currency"],"Service_Risk_Value":fmt["currency"]}, widths={"Description":30})

        pur = wb.add_worksheet("Purchase Plan")
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "CompletePurchasePlan", formats={**fmt,"Unit_Cost":fmt["currency"],"Purchase_Value":fmt["currency"],"Recommended_Order":fmt["integer"],"Open_PO":fmt["integer"]}, widths={"Description":30})

        act = wb.add_worksheet("Action Plan")
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "CompleteActionPlan", formats={**fmt,"Purchase_Value":fmt["currency"]}, widths={"Description":30,"Reason":34})

        sup = wb.add_worksheet("Supplier Risk")
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

        dqs = wb.add_worksheet("Data Quality")
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "CompleteDQ", widths={"Check":30,"Details":42})

        src_ws = wb.add_worksheet("Source Data")
        _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
        _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "CompleteSourceData")


    if comparison is not None and comparison_meta is not None and not comparison.empty:
        cm = wb.add_worksheet("Change Monitor")
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
    pa = wb.add_worksheet("Planning Agent")
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
    lines = [
        "### Prioridades de esta semana",
        "",
        "| Prioridad | SKU | Acción | Timing | Cantidad | Riesgo | Confianza |",
        "|---|---|---|---|---:|---|---|"
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        if r["Action"] == "BUY_NOW":
            action = "Comprar ahora"
        elif r["Action"] == "CONFIRM_PO":
            action = "Confirmar PO"
        elif r["Action"] == "DO_NOT_BUY":
            action = "No comprar"
        elif r["Action"] == "REVIEW":
            action = "Revisar"
        else:
            action = "Monitorizar"

        qty = f"{r['Recommended_Order']:.0f}" if r["Recommended_Order"] > 0 else "—"
        risk = f"{r['Days_Cover']:.1f}d cover / {r['Lead_Time_Days']:.0f}d LT"
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
        f"**Compra recomendada total:** €{purchase_value:,.0f}",
        f"**Inventario actualmente en exceso:** €{excess_value:,.0f}",
    ]

    if not critical.empty:
        lines.append("")
        lines.append("**Acción inmediata:** emitir/validar pedidos de los SKUs `BUY_NOW` y confirmar fecha de entrega con el proveedor.")
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
    lines = ["## Supply Chain Agent — análisis"]
    for tool in tools:
        lines.append(f"### {tool['name']}")
        if tool["name"] == "logistics_kpis":
            k = tool["kpis"]
            lines.append(
                f"Inventario **€{k['inventory_value']:,.0f}** · rotación **{k['inventory_turns']:.2f}x** · "
                f"cobertura agregada **{k['days_cover']:.1f} días** · cobertura de lead time **{k['lead_time_coverage_pct']:.1f}%**."
            )
            lines.append(
                f"Riesgo de servicio **€{k['service_risk_value']:,.0f}** · exceso **€{k['excess_inventory_value']:,.0f}** · "
                f"PO abiertas **€{k['open_po_value']:,.0f}** · compra recomendada **€{k['purchase_requirement']:,.0f}**."
            )
        elif tool["name"] == "purchase_planner":
            k = tool["kpis"]
            lines.append(f"Compra recomendada: **€{k['recommended_purchase_value']:,.0f}** · {k['lines']} líneas · {k['critical_lines']} críticas.")
            lines += [
                f"- **{r['SKU']}** → {r['Action']} · {r['Recommended_Order']:.0f} uds · cover {r['Days_Cover']:.1f}d · LT {r['Lead_Time_Days']:.0f}d."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "inventory_optimizer":
            k = tool["kpis"]
            lines.append(f"Exceso estimado: **€{k['excess_inventory_value']:,.0f}** en {k['sku_count']} SKUs.")
        elif tool["name"] == "service_risk":
            k = tool["kpis"]
            lines.append(f"Exposición de servicio: **€{k['service_risk_value']:,.0f}** en {k['sku_count']} SKUs.")
        elif tool["name"] == "supplier_risk":
            lines += [
                f"- **{r['Supplier']}** → críticos {int(r['Critical'])}, compras €{r['Purchase_Value']:,.0f}, riesgo servicio €{r['Service_Risk_Value']:,.0f}."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "demand_outlook":
            lines.append("Mayores subidas de demanda:")
            lines += [
                f"- **{r['SKU']}** → forecast {r['Forecast_Next_Month']:.0f} uds ({r['Forecast_Change_Pct']:+.1f}%)."
                for r in tool["rising"][:5]
            ]
        elif tool["name"] == "planning_agent":
            k = tool["kpis"]
            lines.append(
                f"Plan de ejecución: **{k['immediate_count']} acciones inmediatas**, "
                f"€{k['purchase_value']:,.0f} de exposición de compra y "
                f"€{k['service_risk_addressed']:,.0f} de riesgo de servicio potencialmente abordable."
            )
            lines += [
                f"- **Paso {r['Execution_Priority']} — {r['SKU']}**: {r['Execution_Task']} · "
                f"{r['Dependency']} · {r['Action_Timing']}."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "change_monitor":
            if not tool["kpis"].get("available"):
                lines.append("No hay dos periodos comparables disponibles.")
            else:
                k = tool["kpis"]
                lines.append(
                    f"Comparación **{k['previous_period']} → {k['current_period']}**: "
                    f"compras {k['purchase_delta']:+,.0f} €, riesgo de servicio {k['service_risk_delta']:+,.0f} €, "
                    f"cambios de acción {k['action_changes']}."
                )
                lines += [
                    f"- **{r['SKU']}** → {r['Change_Classification']} · cobertura {r['Days_Cover_Delta']:+.1f}d · "
                    f"compra {r['Purchase_Value_Delta']:+,.0f} € · {r['Change_Reason']}"
                    for r in tool["rows"][:8]
                ]
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
        return False, "No hay API key configurada.", {}
    if OpenAI is None:
        return False, "La librería OpenAI no está instalada.", {}

    try:
        client = _make_openai_client(api_key)

        # Authentication/permissions test independent from the selected model.
        client.models.list()

        # Small Responses API smoke test.
        response = client.responses.create(
            model=model,
            input="Responde únicamente: conexión OK"
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
                "Eres un agente senior de Supply Chain. Responde en español y de forma operativa. "
                "Usa los resultados estructurados de las herramientas. No inventes números. "
                "No ejecutes compras. Para compras distingue BUY_NOW de CONFIRM_PO. "
                "Si existe PO abierta, confirma su adecuación antes de recomendar una nueva compra. "
                "Para exceso cuantifica el valor de inventario potencialmente liberable. "
                "Para servicio cuantifica unidades y valor expuesto. "
                "Para proveedores explica concentración de riesgo. "
                "Para forecast señala cambios que puedan modificar decisiones. "
                "Para KPIs logísticos usa la herramienta logistics_kpis y distingue claramente cobertura de lead time de OTIF/fill rate. "
                "Para comparación explica qué ha mejorado, empeorado o cambiado de acción entre periodos y cuantifica los deltas. "
                "En preguntas ejecutivas: Resumen → Top 3 prioridades → Acciones → Riesgos/Supuestos."
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
# Session state
# -----------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "copilot_prefill" not in st.session_state:
    st.session_state.copilot_prefill = ""

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
    st.caption("Decision Intelligence for planners · V2.0.6")

    with st.expander("🧭 How to use the Copilot", expanded=True):
        st.markdown("""
**Recommended workflow**

1. **📊 Dashboard** → understand the overall situation.  
2. **🎯 Decision Center** → identify what needs action now.  
3. **🧠 Planning Agent** → follow the recommended execution sequence.  
4. **🤖 Copilot** → ask why, explore alternatives and investigate exceptions.  
5. **📤 Export** → share the management-ready reports.
        """)

    uploaded = st.file_uploader("Histórico de demanda e inventario", type=["csv", "xlsx", "xls"])
    safety_days = st.slider("Safety stock floor (días)", 0, 90, 10)
    service = st.select_slider("Service level", options=[0.90,0.95,0.975,0.99], value=0.95)
    st.subheader("🤖 OpenAI")

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
        "Modelo",
        ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        index=0,
        help="Modelos actuales disponibles en la Responses API."
    )


    if stored_key:
        prefix = stored_key[:8] if len(stored_key) >= 8 else stored_key
        suffix = stored_key[-4:] if len(stored_key) >= 4 else ""
        st.success(f"API key detectada · {source}")
        st.caption(f"Fingerprint: `{prefix}…{suffix}`")
    else:
        st.warning("No hay OPENAI_API_KEY configurada. Se usará el modo local.")

    use_manual = st.checkbox(
        "Probar otra clave solo en esta sesión",
        value=False
    )

    manual_key = ""
    if use_manual:
        manual_key = st.text_input(
            "API key manual",
            type="password"
        ).strip()

    effective_key = manual_key if use_manual and manual_key else stored_key

    if st.button("🔌 Diagnosticar conexión OpenAI", use_container_width=True):
        ok, message, details = test_openai_connection(effective_key, model)

        if ok:
            st.success(f"✅ OpenAI conectado: {message}")
        else:
            st.error("❌ No se pudo validar la conexión.")
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
        "La API key nunca se muestra completa ni se guarda en GitHub."
    )

    if st.button("🔄 Cargar demo"):
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
        summary = wb.add_worksheet("Summary")
        summary.hide_gridlines(2)
        summary.write(0, 0, title, title_fmt)
        summary.write(1, 0, "Exported from Supply Chain AI Copilot V2.0.6", subtitle_fmt)
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

K = kpis(a)
logistics_dashboard = build_logistics_dashboard(a, raw)
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
    st.markdown("### 📁 Current dataset")
    st.caption(f"{len(raw):,} rows · {a['SKU'].nunique():,} SKUs · {a['Supplier'].nunique():,} suppliers")
    if available_periods:
        st.caption(f"Period: **{available_periods[0]} → {available_periods[-1]}**")
    if K["critical"] > 0:
        st.warning(f"{K['critical']:,} critical SKUs require attention.")
    else:
        st.success("No critical SKUs under the current planning parameters.")

    with st.expander("⚙️ Planning assumptions"):
        st.caption(f"Safety stock floor: **{safety_days} days**")
        st.caption(f"Service level target: **{service:.1%}**")
        st.caption("These parameters affect safety stock, coverage and purchase recommendations.")

    if len(available_periods) >= 2:
        with st.expander("🔄 Comparación de periodos", expanded=False):
            selected_current = st.selectbox(
                "Periodo actual",
                available_periods,
                index=available_periods.index(default_current),
                key="change_current_period"
            )
            current_idx = available_periods.index(selected_current)
            prev_options = available_periods[:current_idx] or [available_periods[0]]
            selected_previous = st.selectbox(
                "Comparar con",
                prev_options,
                index=prev_options.index(default_previous) if default_previous in prev_options else len(prev_options)-1,
                key="change_previous_period"
            )
            st.caption(f"Actual: {selected_current} · Anterior: {selected_previous}")

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
st.caption("From raw supply-chain data to prioritized decisions · V2.0.6")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SKUs", K["sku"])
c2.metric("🔴 Critical", K["critical"])
c3.metric("🟠 Review", K["review"])
c4.metric("🛒 Purchase need", f"€{K['purchase']:,.0f}")
c5.metric("💰 Inventory", f"€{K['inventory']:,.0f}")
c6.metric("📈 Next month", f"{K['forecast']:,.0f}")

st.markdown("""
<div class="sc-workflow">
<span class="sc-step">1️⃣ Understand</span> <span class="sc-muted">Dashboard</span>
→ <span class="sc-step">2️⃣ Prioritize</span> <span class="sc-muted">Decision Center</span>
→ <span class="sc-step">3️⃣ Execute</span> <span class="sc-muted">Planning Agent</span>
→ <span class="sc-step">4️⃣ Investigate</span> <span class="sc-muted">Copilot</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🎯 Decision Center","📊 Logistics Dashboard","🧠 Planning Agent","📊 Inventory","📈 Forecast",
    "🧩 ABC/XYZ","🚚 Suppliers","🧪 Scenarios","🧹 Data Quality",
    "📝 Action Plan","🔄 Change Monitor","🤖 Copilot","📤 Export"
])

# -----------------------------
# Decision Center
# -----------------------------
with tabs[0]:
    st.subheader("What should the planner do now?")
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

    st.subheader("Why these actions?")
    st.info(
        "The engine treats a SKU as BUY_NOW when on-hand stock is below lead-time demand. "
        "REVIEW/CONFIRM_PO cases are handled separately to avoid double ordering when an open PO already exists. "
        "DO_NOT_BUY cases are flagged when coverage is materially above the policy threshold."
    )

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

    st.subheader("Purchase plan by supplier")
    po = export_purchase(a)
    if po.empty:
        st.success("No purchase orders recommended.")
    else:
        st.dataframe(po, use_container_width=True, hide_index=True)

    st.markdown("#### 🤖 Continue with Copilot")
    dcq1, dcq2, dcq3 = st.columns(3)
    if dcq1.button("🔴 Explain critical SKUs", use_container_width=True, key="dc_critical_copilot"):
        st.session_state.copilot_prefill = "Explica los 5 SKUs críticos más importantes, qué está provocando el riesgo y qué acción debería revisar primero."
    if dcq2.button("🛒 Explain purchase plan", use_container_width=True, key="dc_purchase_copilot"):
        st.session_state.copilot_prefill = "Explica el purchase plan actual, qué proveedores concentran más valor y cuáles son las prioridades de compra."
    if dcq3.button("⚠️ Explain service risk", use_container_width=True, key="dc_service_copilot"):
        st.session_state.copilot_prefill = "Explica dónde está concentrado el service risk y qué acciones podrían reducirlo sin generar compras innecesarias."
    if st.session_state.get("copilot_prefill"):
        st.info("Pregunta preparada para Copilot. Ve a la pestaña 🤖 Copilot para ejecutarla.")

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
    st.subheader("📊 Logistics KPI Dashboard")
    st.caption("Executive view of inventory, service exposure, replenishment, supplier exposure and logistics efficiency.")
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
    r3[3].metric("📈 Next-month forecast", f"{a['Forecast_Next_Month'].sum():,.0f} units")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Sales value vs inventory value")
        if not d["trend"].empty:
            trend_chart = d["trend"].set_index("Period")[["Sales_Value", "Inventory_Value"]]
            st.line_chart(trend_chart, use_container_width=True)
        else:
            st.info("No monthly history available for the trend chart.")
    with c2:
        st.markdown("#### 📦 Inventory health")
        status_chart = d["status"].set_index("Status")[["SKUs"]]
        st.bar_chart(status_chart, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🚚 Supplier exposure")
        supplier_chart = d["supplier"].head(10).set_index("Supplier")[["Inventory_Value", "Service_Risk_Value", "Purchase_Value"]]
        st.bar_chart(supplier_chart, use_container_width=True)
    with c4:
        st.markdown("#### ⏱️ Lead-time profile")
        lead_chart = d["lead_bins"].set_index("Lead_Time_Bucket")[["SKUs"]]
        st.bar_chart(lead_chart, use_container_width=True)

    st.markdown("#### 🧩 ABC / XYZ portfolio")
    abc_display = d["abc_xyz"].set_index("ABC_Class")
    st.dataframe(abc_display, use_container_width=True)

    st.markdown("#### 📋 Logistics KPI catalogue")
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

    st.markdown("#### 🔎 What the dashboard is telling the planner")
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

    st.markdown("#### 🤖 Investigate with Copilot")
    kq1, kq2, kq3 = st.columns(3)
    if kq1.button("📊 Explain KPI health", use_container_width=True, key="kpi_health_copilot"):
        st.session_state.copilot_prefill = "Analiza la salud de los principales KPI logísticos, identifica las señales que requieren atención y explica sus causas."
    if kq2.button("🚚 Supplier exposure", use_container_width=True, key="kpi_supplier_copilot"):
        st.session_state.copilot_prefill = "Analiza la exposición por proveedor y dime dónde debería concentrar la atención del planner."
    if kq3.button("💰 Working capital", use_container_width=True, key="kpi_wc_copilot"):
        st.session_state.copilot_prefill = "Analiza inventario, exceso, cobertura y compras recomendadas desde la perspectiva de working capital."
    if st.session_state.get("copilot_prefill"):
        st.info("Pregunta preparada para Copilot. Ve a la pestaña 🤖 Copilot para ejecutarla.")

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
    st.caption("Service and coverage KPIs are planning proxies derived from inventory, demand and lead-time data; they are not OTIF or customer fill-rate measurements unless those source fields are provided.")

# -----------------------------
# Planning Agent
# -----------------------------
with tabs[2]:
    st.subheader("🧠 Planning Agent")
    st.caption("Turns the decision engine into an ordered sequence of planner actions.")

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

    st.subheader("Recommended execution sequence")
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

    st.subheader("Planner rationale")
    selected_plan_sku = st.selectbox(
        "Explain the recommended action for",
        options=[""] + planning["SKU"].astype(str).tolist()
    )
    if selected_plan_sku:
        r = planning[planning["SKU"].astype(str) == selected_plan_sku].iloc[0]
        st.info(r["Planning_Rationale"])

    st.subheader("Export Planning Agent")
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
            st.warning(f"Excel export unavailable: {planning_exc}")
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
    st.subheader("Inventory health")
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
    st.subheader("Demand outlook")
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
    st.info(
        "El forecast del MVP utiliza una media ponderada de los últimos 6 meses más una tendencia lineal. "
        "La siguiente iteración puede añadir estacionalidad, demanda intermitente y modelos alternativos."
    )

# -----------------------------
# ABC/XYZ
# -----------------------------
with tabs[5]:
    st.subheader("Segmentation")
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
    st.subheader("Supplier exposure")
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
with tabs[7]:
    st.subheader("🧪 Policy simulator")
    st.caption("Simula decisiones antes de cambiar la política.")
    s1, s2 = st.columns(2)
    with s1:
        sim_safety = st.slider("Safety stock floor", 0, 90, safety_days, key="sim_safety")
    with s2:
        sim_lead = st.slider("Lead time multiplier", .5, 2.0, 1.0, .05, key="sim_lead")

    sim = raw.copy()
    sim["Lead_Time_Days"] = pd.to_numeric(sim["Lead_Time_Days"], errors="coerce").fillna(0)*sim_lead
    sim_a = analyze(sim, sim_safety, service)
    base_val = float(a["Purchase_Value"].sum())
    sim_val = float(sim_a["Purchase_Value"].sum())
    base_risk = K["critical"]
    sim_risk = int((sim_a["Status"]=="🔴 CRITICAL").sum())
    m1,m2,m3 = st.columns(3)
    m1.metric("Purchase need", f"€{sim_val:,.0f}", f"{sim_val-base_val:+,.0f}")
    m2.metric("Critical SKUs", sim_risk, f"{sim_risk-base_risk:+d}")
    m3.metric("Required stock value", f"€{(sim_a['Required_Stock']*sim_a['Unit_Cost']).sum():,.0f}")


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
    ws = wb.add_worksheet("Summary")
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

    detail = wb.add_worksheet("Quality Checks")
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
# Data Quality
# -----------------------------
with tabs[8]:
    st.subheader("🧹 Data Quality")
    st.caption("Checks the data before operational decisions are used.")

    dq_summary = data_quality_summary(raw, dq)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", dq_summary["rows"])
    q2.metric("SKUs", dq_summary["skus"])
    q3.metric("Suppliers", dq_summary["suppliers"])
    q4.metric("Warnings", dq_summary["warnings"])
    q5.metric("Critical", dq_summary["critical"])

    if dq_summary["critical"] > 0:
        st.error("⛔ Critical data-quality issues detected.")
    elif dq_summary["warnings"] > 0:
        st.warning("⚠️ Data-quality warnings detected. Review them before issuing purchase decisions.")
    else:
        st.success("✅ All current data-quality checks passed.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Latest period", dq_summary["latest_period"])
        st.metric("Checks completed", dq_summary["checks"])
    with c2:
        st.dataframe(
            dq[["Category","Check","Status","Count","Details"]],
            use_container_width=True, hide_index=True
        )

    st.subheader("📤 Export Data Quality")
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
            st.warning(f"Excel export unavailable: {dq_exc}")
        if dq_excel:
            st.download_button(
                "📗 Data Quality Report (Excel)",
                dq_excel,
                "data_quality_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="data_quality_excel_export"
            )
    st.caption("HTML provides the management-ready visual report; Excel provides editable quality checks and KPI summary.")


# -----------------------------
# Action Plan
# -----------------------------
with tabs[9]:
    st.subheader("📝 Weekly Action Plan")
    st.caption("Planner-ready worklist generated by the Decision Engine.")

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

    st.subheader("Supplier follow-up")
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
    st.markdown("### 📤 Export Action Plan")
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
    st.caption("HTML y Excel utilizan la vista filtrada actual. El HTML incluye KPIs y una presentación ejecutiva; el Excel incluye Summary, Action Plan y Supplier Summary con filtros.")

# -----------------------------
# Change Monitor
# -----------------------------
with tabs[10]:
    st.subheader("🔄 What changed?")
    if not comparison_meta.get("has_comparison"):
        st.info("Necesitas al menos dos periodos históricos para comparar la evolución.")
    else:
        st.caption(f"Comparando **{comparison_meta['previous_period']} → {comparison_meta['current_period']}**")

        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("Purchase need", f"€{comparison_meta['purchase_current']:,.0f}", f"{comparison_meta['purchase_delta']:+,.0f} €")
        cm2.metric("Service risk", f"€{comparison_meta['service_risk_current']:,.0f}", f"{comparison_meta['service_risk_delta']:+,.0f} €")
        cm3.metric("Excess exposure", f"€{comparison_meta['excess_current']:,.0f}", f"{comparison_meta['excess_delta']:+,.0f} €")
        cm4.metric("Critical SKUs", comparison_meta["critical_current"], f"{comparison_meta['critical_delta']:+d}")
        cm5.metric("Action changes", comparison_meta["action_changes"], f"{comparison_meta['worsened']} worsened")

        if comparison_meta["worsened"] > comparison_meta["improved"]:
            st.warning(f"Hay más cambios desfavorables ({comparison_meta['worsened']}) que favorables ({comparison_meta['improved']}).")
        elif comparison_meta["improved"] > comparison_meta["worsened"]:
            st.success(f"La evolución es mayoritariamente favorable: {comparison_meta['improved']} mejorados frente a {comparison_meta['worsened']} empeorados.")
        else:
            st.info("La evolución está equilibrada entre mejoras y empeoramientos.")

        view = comparison[[
            "Priority","SKU","Description","Supplier","Previous_Action","Current_Action",
            "Action_Transition","Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
            "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta","Sales_Delta_Pct",
            "Change_Classification","Change_Reason"
        ]].copy()
        st.dataframe(view, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top worsened / watch")
            worsened = comparison[comparison["Change_Classification"].isin(["WORSENED","WATCH"])].head(8)
            st.dataframe(
                worsened[["SKU","Description","Current_Action","Days_Cover_Delta",
                          "Purchase_Value_Delta","Service_Risk_Delta","Change_Classification"]],
                use_container_width=True, hide_index=True
            )
        with c2:
            st.subheader("Top improved")
            improved = comparison[comparison["Change_Classification"]=="IMPROVED"].sort_values(
                ["Service_Risk_Delta","Purchase_Value_Delta"], ascending=[True,True]
            ).head(8)
            st.dataframe(
                improved[["SKU","Description","Previous_Action","Current_Action",
                          "Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta"]],
                use_container_width=True, hide_index=True
            )

        change_report_html = build_change_monitor_html(comparison, comparison_meta)

        st.markdown("### 📤 Export Change Monitor")
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
            st.warning(f"Excel export unavailable: {export_exc}")

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
with tabs[11]:
    st.subheader("🤖 Your Supply Chain Copilot")
    if st.session_state.get("copilot_prefill"):
        pending = st.session_state["copilot_prefill"]
        st.info(f"**Prepared analysis:** {pending}")
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
        quick_question = "¿Qué debería comprar esta semana y cuáles son las 3 prioridades más importantes?"
    if q2.button("💰 Reduce inventory"):
        quick_question = "¿Dónde puedo reducir inventario sin aumentar demasiado el riesgo de servicio?"
    if q3.button("🚚 Supplier risk"):
        quick_question = "¿Qué proveedores requieren más atención y por qué?"
    if q4.button("📈 Demand outlook"):
        quick_question = "¿Qué cambios de demanda pueden cambiar mis decisiones de compra?"
    if q5.button("🔄 What changed?"):
        quick_question = "¿Qué ha cambiado entre el último periodo y el anterior y cuáles son las 3 mayores variaciones?"
    if q6.button("📊 Logistics KPIs"):
        quick_question = "¿Cuál es el estado de los principales KPI logísticos y cuáles requieren atención?"
    if quick_question:
        st.session_state.chat.append({"role": "user", "content": quick_question})
        with st.chat_message("user"):
            st.markdown(quick_question)
        with st.chat_message("assistant"):
            with st.spinner("Ejecutando análisis de Supply Chain..."):
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
with tabs[12]:
    st.subheader("📤 Reporting Center")
    st.caption("Visual HTML reports and professional Excel workbooks containing the same decision-ready information.")

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

    st.markdown("### 2. Detailed visual reports")
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
            st.warning(f"Excel export unavailable: {planning_export_exc}")
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
        st.warning(f"Excel export unavailable: {excel_error}")

    st.info("Raw CSV exports remain removed from the reporting workflow. HTML and Excel are now the primary shareable outputs.")


st.divider()
st.caption("Supply Chain AI Copilot V2.0.6 — recommendations require planner validation before execution.")

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
    return (
        f"Subject: Urgent supply chain follow-up — {row['SKU']} / {row['Description']}\n\n"
        f"Hello {row['Supplier']} team,\n\n"
        f"We are reviewing replenishment for {row['SKU']} ({row['Description']}). "
        f"The current stock coverage is {row['Days_Cover']:.1f} days and the lead time is "
        f"{row['Lead_Time_Days']:.0f} days.\n\n"
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

def _html_shell(title, subtitle, body):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
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
<div class="meta">Generated {generated} · Supply Chain AI Copilot V2.0.5</div>
</div>
{body}
<div class="footer">Decision support only. Validate purchase execution and supplier commitments before release.</div>
</div>
</body>
</html>"""

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
            '<div class="section"><div class="note">No hay dos periodos comparables disponibles.</div></div>'
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

    for j, col in enumerate(columns):
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
            "columns": [{"header": c} for c in columns],
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

    ws = wb.add_worksheet("Executive")
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

    action_ws = wb.add_worksheet("Action Plan")
    _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
    _xlsx_write_df(
        action_ws, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "ExecutiveActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    dq_ws = wb.add_worksheet("Data Quality")
    _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecutiveDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_detailed(a, raw, dq, plan):
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)

    inv = wb.add_worksheet("Inventory Risk")
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

    pur = wb.add_worksheet("Purchase Plan")
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

    act = wb.add_worksheet("Action Plan")
    _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
    _xlsx_write_df(
        act, plan, 3, 0,
        ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
        wb, "DetailedActionPlan",
        formats={**fmt, "Purchase_Value": fmt["currency"]},
        widths={"Description": 30, "Reason": 34}
    )

    sup = wb.add_worksheet("Supplier Risk")
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

    dq_ws = wb.add_worksheet("Data Quality")
    _xlsx_title(dq_ws, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
    _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDataQuality",
                   widths={"Check": 30, "Details": 42})
    return wb

def _xlsx_build_complete(a, raw, dq, plan):
    wb = _xlsx_build_detailed(a, raw, dq, plan)

    # Complete pack adds Executive + Source Data to the detailed workbook.
    ex = wb.add_worksheet("Executive")
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

    src_ws = wb.add_worksheet("Source Data")
    _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
    _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "SourceData")

    return wb



def _excel_planning_agent_bytes(planning, meta):
    if xlsxwriter is None:
        raise RuntimeError("XlsxWriter is not available. Add XlsxWriter to requirements.txt and redeploy.")
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    fmt = _xlsx_base_formats(wb)
    ws = wb.add_worksheet("Planning Agent")

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
    ws = wb.add_worksheet("Change Monitor")

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
        ws = wb.add_worksheet("Executive")
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
        action_ws = wb.add_worksheet("Action Plan")
        _xlsx_title(action_ws, "Weekly Action Plan", "Planner-ready worklist.", wb, 9)
        _xlsx_write_df(action_ws, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "ExecActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        dq_ws = wb.add_worksheet("Data Quality")
        _xlsx_title(dq_ws, "Data Quality", "Structural and consistency checks.", wb, 4)
        _xlsx_write_df(dq_ws, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "ExecDQ", widths={"Check":30,"Details":42})

    elif kind == "detailed":
        fmt = _xlsx_base_formats(wb)
        inv = wb.add_worksheet("Inventory Risk")
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "InventoryRisk", formats={**fmt, "Excess_Inventory_Value": fmt["currency"], "Service_Risk_Value": fmt["currency"]}, widths={"Description":30})
        _xlsx_write_section_chart(inv, wb, "bar", "Excess inventory value", 0, 9, 4, min(3+len(x),13), "M4")
        pur = wb.add_worksheet("Purchase Plan")
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "PurchasePlan", formats={**fmt, "Unit_Cost": fmt["currency"], "Purchase_Value": fmt["currency"], "Recommended_Order":fmt["integer"], "Open_PO":fmt["integer"]}, widths={"Description":30})
        _xlsx_write_section_chart(pur, wb, "column", "Purchase value by SKU", 0, 6, 4, min(3+len(x),13), "M4")
        act = wb.add_worksheet("Action Plan")
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "DetailedActionPlan", formats={**fmt, "Purchase_Value": fmt["currency"]}, widths={"Description":30,"Reason":34})
        sup = wb.add_worksheet("Supplier Risk")
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
        dqs = wb.add_worksheet("Data Quality")
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "DetailedDQ", widths={"Check":30,"Details":42})

    else:
        fmt = _xlsx_base_formats(wb)
        ex = wb.add_worksheet("Executive")
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
        inv = wb.add_worksheet("Inventory Risk")
        _xlsx_title(inv, "Inventory & Service Risk", "Excess inventory and service-risk exposure.", wb, 10)
        x = a.sort_values("Excess_Inventory_Value", ascending=False).head(20)
        _xlsx_write_df(inv, x, 3, 0,
                       ["SKU","Description","Supplier","Status","Days_Cover","Lead_Time_Days","Stock","Open_PO","Excess_Inventory_Qty","Excess_Inventory_Value","Service_Risk_Value"],
                       wb, "CompleteInventoryRisk", formats={**fmt,"Excess_Inventory_Value":fmt["currency"],"Service_Risk_Value":fmt["currency"]}, widths={"Description":30})

        pur = wb.add_worksheet("Purchase Plan")
        _xlsx_title(pur, "Purchase Plan", "Recommended replenishment by SKU and supplier.", wb, 10)
        x = a[a["Recommended_Order"] > 0].sort_values("Purchase_Value", ascending=False)
        _xlsx_write_df(pur, x, 3, 0,
                       ["SKU","Description","Supplier","Action","Recommended_Order","Unit_Cost","Purchase_Value","Days_Cover","Lead_Time_Days","Open_PO","PO_Adequacy"],
                       wb, "CompletePurchasePlan", formats={**fmt,"Unit_Cost":fmt["currency"],"Purchase_Value":fmt["currency"],"Recommended_Order":fmt["integer"],"Open_PO":fmt["integer"]}, widths={"Description":30})

        act = wb.add_worksheet("Action Plan")
        _xlsx_title(act, "Weekly Action Plan", "Owner, timing, reason and confidence.", wb, 9)
        _xlsx_write_df(act, plan, 3, 0,
                       ["Priority","SKU","Description","Supplier","Action","Owner","Deadline","Reason","Confidence","Purchase_Value"],
                       wb, "CompleteActionPlan", formats={**fmt,"Purchase_Value":fmt["currency"]}, widths={"Description":30,"Reason":34})

        sup = wb.add_worksheet("Supplier Risk")
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

        dqs = wb.add_worksheet("Data Quality")
        _xlsx_title(dqs, "Data Quality", "Severity and counts for the current dataset.", wb, 4)
        _xlsx_write_df(dqs, dq, 3, 0, ["Category","Check","Status","Count","Details"], wb, "CompleteDQ", widths={"Check":30,"Details":42})

        src_ws = wb.add_worksheet("Source Data")
        _xlsx_title(src_ws, "Source Data", "Normalized source dataset used by the decision engine.", wb, max(5, len(raw.columns)-1))
        _xlsx_write_df(src_ws, raw, 3, 0, list(raw.columns), wb, "CompleteSourceData")


    if comparison is not None and comparison_meta is not None and not comparison.empty:
        cm = wb.add_worksheet("Change Monitor")
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
    pa = wb.add_worksheet("Planning Agent")
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
    lines = [
        "### Prioridades de esta semana",
        "",
        "| Prioridad | SKU | Acción | Timing | Cantidad | Riesgo | Confianza |",
        "|---|---|---|---|---:|---|---|"
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        if r["Action"] == "BUY_NOW":
            action = "Comprar ahora"
        elif r["Action"] == "CONFIRM_PO":
            action = "Confirmar PO"
        elif r["Action"] == "DO_NOT_BUY":
            action = "No comprar"
        elif r["Action"] == "REVIEW":
            action = "Revisar"
        else:
            action = "Monitorizar"

        qty = f"{r['Recommended_Order']:.0f}" if r["Recommended_Order"] > 0 else "—"
        risk = f"{r['Days_Cover']:.1f}d cover / {r['Lead_Time_Days']:.0f}d LT"
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
        f"**Compra recomendada total:** €{purchase_value:,.0f}",
        f"**Inventario actualmente en exceso:** €{excess_value:,.0f}",
    ]

    if not critical.empty:
        lines.append("")
        lines.append("**Acción inmediata:** emitir/validar pedidos de los SKUs `BUY_NOW` y confirmar fecha de entrega con el proveedor.")
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
    lines = ["## Supply Chain Agent — análisis"]
    for tool in tools:
        lines.append(f"### {tool['name']}")
        if tool["name"] == "logistics_kpis":
            k = tool["kpis"]
            lines.append(
                f"Inventario **€{k['inventory_value']:,.0f}** · rotación **{k['inventory_turns']:.2f}x** · "
                f"cobertura agregada **{k['days_cover']:.1f} días** · cobertura de lead time **{k['lead_time_coverage_pct']:.1f}%**."
            )
            lines.append(
                f"Riesgo de servicio **€{k['service_risk_value']:,.0f}** · exceso **€{k['excess_inventory_value']:,.0f}** · "
                f"PO abiertas **€{k['open_po_value']:,.0f}** · compra recomendada **€{k['purchase_requirement']:,.0f}**."
            )
        elif tool["name"] == "purchase_planner":
            k = tool["kpis"]
            lines.append(f"Compra recomendada: **€{k['recommended_purchase_value']:,.0f}** · {k['lines']} líneas · {k['critical_lines']} críticas.")
            lines += [
                f"- **{r['SKU']}** → {r['Action']} · {r['Recommended_Order']:.0f} uds · cover {r['Days_Cover']:.1f}d · LT {r['Lead_Time_Days']:.0f}d."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "inventory_optimizer":
            k = tool["kpis"]
            lines.append(f"Exceso estimado: **€{k['excess_inventory_value']:,.0f}** en {k['sku_count']} SKUs.")
        elif tool["name"] == "service_risk":
            k = tool["kpis"]
            lines.append(f"Exposición de servicio: **€{k['service_risk_value']:,.0f}** en {k['sku_count']} SKUs.")
        elif tool["name"] == "supplier_risk":
            lines += [
                f"- **{r['Supplier']}** → críticos {int(r['Critical'])}, compras €{r['Purchase_Value']:,.0f}, riesgo servicio €{r['Service_Risk_Value']:,.0f}."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "demand_outlook":
            lines.append("Mayores subidas de demanda:")
            lines += [
                f"- **{r['SKU']}** → forecast {r['Forecast_Next_Month']:.0f} uds ({r['Forecast_Change_Pct']:+.1f}%)."
                for r in tool["rising"][:5]
            ]
        elif tool["name"] == "planning_agent":
            k = tool["kpis"]
            lines.append(
                f"Plan de ejecución: **{k['immediate_count']} acciones inmediatas**, "
                f"€{k['purchase_value']:,.0f} de exposición de compra y "
                f"€{k['service_risk_addressed']:,.0f} de riesgo de servicio potencialmente abordable."
            )
            lines += [
                f"- **Paso {r['Execution_Priority']} — {r['SKU']}**: {r['Execution_Task']} · "
                f"{r['Dependency']} · {r['Action_Timing']}."
                for r in tool["rows"][:8]
            ]
        elif tool["name"] == "change_monitor":
            if not tool["kpis"].get("available"):
                lines.append("No hay dos periodos comparables disponibles.")
            else:
                k = tool["kpis"]
                lines.append(
                    f"Comparación **{k['previous_period']} → {k['current_period']}**: "
                    f"compras {k['purchase_delta']:+,.0f} €, riesgo de servicio {k['service_risk_delta']:+,.0f} €, "
                    f"cambios de acción {k['action_changes']}."
                )
                lines += [
                    f"- **{r['SKU']}** → {r['Change_Classification']} · cobertura {r['Days_Cover_Delta']:+.1f}d · "
                    f"compra {r['Purchase_Value_Delta']:+,.0f} € · {r['Change_Reason']}"
                    for r in tool["rows"][:8]
                ]
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
        return False, "No hay API key configurada.", {}
    if OpenAI is None:
        return False, "La librería OpenAI no está instalada.", {}

    try:
        client = _make_openai_client(api_key)

        # Authentication/permissions test independent from the selected model.
        client.models.list()

        # Small Responses API smoke test.
        response = client.responses.create(
            model=model,
            input="Responde únicamente: conexión OK"
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
                "Eres un agente senior de Supply Chain. Responde en español y de forma operativa. "
                "Usa los resultados estructurados de las herramientas. No inventes números. "
                "No ejecutes compras. Para compras distingue BUY_NOW de CONFIRM_PO. "
                "Si existe PO abierta, confirma su adecuación antes de recomendar una nueva compra. "
                "Para exceso cuantifica el valor de inventario potencialmente liberable. "
                "Para servicio cuantifica unidades y valor expuesto. "
                "Para proveedores explica concentración de riesgo. "
                "Para forecast señala cambios que puedan modificar decisiones. "
                "Para KPIs logísticos usa la herramienta logistics_kpis y distingue claramente cobertura de lead time de OTIF/fill rate. "
                "Para comparación explica qué ha mejorado, empeorado o cambiado de acción entre periodos y cuantifica los deltas. "
                "En preguntas ejecutivas: Resumen → Top 3 prioridades → Acciones → Riesgos/Supuestos."
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
# Session state
# -----------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📦 Supply Chain AI")
    st.caption("Decision Intelligence for planners")
    uploaded = st.file_uploader("Histórico de demanda e inventario", type=["csv", "xlsx", "xls"])
    safety_days = st.slider("Safety stock floor (días)", 0, 90, 10)
    service = st.select_slider("Service level", options=[0.90,0.95,0.975,0.99], value=0.95)
    st.subheader("🤖 OpenAI")

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
        "Modelo",
        ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        index=0,
        help="Modelos actuales disponibles en la Responses API."
    )


    if stored_key:
        prefix = stored_key[:8] if len(stored_key) >= 8 else stored_key
        suffix = stored_key[-4:] if len(stored_key) >= 4 else ""
        st.success(f"API key detectada · {source}")
        st.caption(f"Fingerprint: `{prefix}…{suffix}`")
    else:
        st.warning("No hay OPENAI_API_KEY configurada. Se usará el modo local.")

    use_manual = st.checkbox(
        "Probar otra clave solo en esta sesión",
        value=False
    )

    manual_key = ""
    if use_manual:
        manual_key = st.text_input(
            "API key manual",
            type="password"
        ).strip()

    effective_key = manual_key if use_manual and manual_key else stored_key

    if st.button("🔌 Diagnosticar conexión OpenAI", use_container_width=True):
        ok, message, details = test_openai_connection(effective_key, model)

        if ok:
            st.success(f"✅ OpenAI conectado: {message}")
        else:
            st.error("❌ No se pudo validar la conexión.")
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
        "La API key nunca se muestra completa ni se guarda en GitHub."
    )

    if st.button("🔄 Cargar demo"):
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
        summary = wb.add_worksheet("Summary")
        summary.hide_gridlines(2)
        summary.write(0, 0, title, title_fmt)
        summary.write(1, 0, "Exported from Supply Chain AI Copilot V2.0.5", subtitle_fmt)
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

K = kpis(a)
logistics_dashboard = build_logistics_dashboard(a, raw)
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
    if len(available_periods) >= 2:
        with st.expander("🔄 Comparación de periodos"):
            selected_current = st.selectbox(
                "Periodo actual",
                available_periods,
                index=available_periods.index(default_current),
                key="change_current_period"
            )
            current_idx = available_periods.index(selected_current)
            prev_options = available_periods[:current_idx] or [available_periods[0]]
            selected_previous = st.selectbox(
                "Comparar con",
                prev_options,
                index=prev_options.index(default_previous) if default_previous in prev_options else len(prev_options)-1,
                key="change_previous_period"
            )
            st.caption(f"Actual: {selected_current} · Anterior: {selected_previous}")

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
st.caption("From raw supply-chain data to prioritized decisions · V2.0.5")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SKUs", K["sku"])
c2.metric("🔴 Critical", K["critical"])
c3.metric("🟠 Review", K["review"])
c4.metric("🛒 Purchase need", f"€{K['purchase']:,.0f}")
c5.metric("💰 Inventory", f"€{K['inventory']:,.0f}")
c6.metric("📈 Next month", f"{K['forecast']:,.0f}")

tabs = st.tabs([
    "🎯 Decision Center","📊 Logistics Dashboard","🧠 Planning Agent","📊 Inventory","📈 Forecast",
    "🧩 ABC/XYZ","🚚 Suppliers","🧪 Scenarios","🧹 Data Quality",
    "📝 Action Plan","🔄 Change Monitor","🤖 Copilot","📤 Export"
])

# -----------------------------
# Decision Center
# -----------------------------
with tabs[0]:
    st.subheader("What should the planner do now?")
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

    st.subheader("Why these actions?")
    st.info(
        "The engine treats a SKU as BUY_NOW when on-hand stock is below lead-time demand. "
        "REVIEW/CONFIRM_PO cases are handled separately to avoid double ordering when an open PO already exists. "
        "DO_NOT_BUY cases are flagged when coverage is materially above the policy threshold."
    )

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

    st.subheader("Purchase plan by supplier")
    po = export_purchase(a)
    if po.empty:
        st.success("No purchase orders recommended.")
    else:
        st.dataframe(po, use_container_width=True, hide_index=True)

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
    st.subheader("📊 Logistics KPI Dashboard")
    st.caption("Executive view of inventory, service exposure, replenishment, supplier exposure and logistics efficiency.")
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
    r3[3].metric("📈 Next-month forecast", f"{a['Forecast_Next_Month'].sum():,.0f} units")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Sales value vs inventory value")
        if not d["trend"].empty:
            trend_chart = d["trend"].set_index("Period")[["Sales_Value", "Inventory_Value"]]
            st.line_chart(trend_chart, use_container_width=True)
        else:
            st.info("No monthly history available for the trend chart.")
    with c2:
        st.markdown("#### 📦 Inventory health")
        status_chart = d["status"].set_index("Status")[["SKUs"]]
        st.bar_chart(status_chart, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🚚 Supplier exposure")
        supplier_chart = d["supplier"].head(10).set_index("Supplier")[["Inventory_Value", "Service_Risk_Value", "Purchase_Value"]]
        st.bar_chart(supplier_chart, use_container_width=True)
    with c4:
        st.markdown("#### ⏱️ Lead-time profile")
        lead_chart = d["lead_bins"].set_index("Lead_Time_Bucket")[["SKUs"]]
        st.bar_chart(lead_chart, use_container_width=True)

    st.markdown("#### 🧩 ABC / XYZ portfolio")
    abc_display = d["abc_xyz"].set_index("ABC_Class")
    st.dataframe(abc_display, use_container_width=True)

    st.markdown("#### 📋 Logistics KPI catalogue")
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

    st.markdown("#### 🔎 What the dashboard is telling the planner")
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
    st.caption("Service and coverage KPIs are planning proxies derived from inventory, demand and lead-time data; they are not OTIF or customer fill-rate measurements unless those source fields are provided.")

# -----------------------------
# Planning Agent
# -----------------------------
with tabs[2]:
    st.subheader("🧠 Planning Agent")
    st.caption("Turns the decision engine into an ordered sequence of planner actions.")

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

    st.subheader("Recommended execution sequence")
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

    st.subheader("Planner rationale")
    selected_plan_sku = st.selectbox(
        "Explain the recommended action for",
        options=[""] + planning["SKU"].astype(str).tolist()
    )
    if selected_plan_sku:
        r = planning[planning["SKU"].astype(str) == selected_plan_sku].iloc[0]
        st.info(r["Planning_Rationale"])

    st.subheader("Export Planning Agent")
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
            st.warning(f"Excel export unavailable: {planning_exc}")
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
    st.subheader("Inventory health")
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
    st.subheader("Demand outlook")
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
    st.info(
        "El forecast del MVP utiliza una media ponderada de los últimos 6 meses más una tendencia lineal. "
        "La siguiente iteración puede añadir estacionalidad, demanda intermitente y modelos alternativos."
    )

# -----------------------------
# ABC/XYZ
# -----------------------------
with tabs[5]:
    st.subheader("Segmentation")
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
    st.subheader("Supplier exposure")
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
with tabs[7]:
    st.subheader("🧪 Policy simulator")
    st.caption("Simula decisiones antes de cambiar la política.")
    s1, s2 = st.columns(2)
    with s1:
        sim_safety = st.slider("Safety stock floor", 0, 90, safety_days, key="sim_safety")
    with s2:
        sim_lead = st.slider("Lead time multiplier", .5, 2.0, 1.0, .05, key="sim_lead")

    sim = raw.copy()
    sim["Lead_Time_Days"] = pd.to_numeric(sim["Lead_Time_Days"], errors="coerce").fillna(0)*sim_lead
    sim_a = analyze(sim, sim_safety, service)
    base_val = float(a["Purchase_Value"].sum())
    sim_val = float(sim_a["Purchase_Value"].sum())
    base_risk = K["critical"]
    sim_risk = int((sim_a["Status"]=="🔴 CRITICAL").sum())
    m1,m2,m3 = st.columns(3)
    m1.metric("Purchase need", f"€{sim_val:,.0f}", f"{sim_val-base_val:+,.0f}")
    m2.metric("Critical SKUs", sim_risk, f"{sim_risk-base_risk:+d}")
    m3.metric("Required stock value", f"€{(sim_a['Required_Stock']*sim_a['Unit_Cost']).sum():,.0f}")


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
    ws = wb.add_worksheet("Summary")
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

    detail = wb.add_worksheet("Quality Checks")
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
# Data Quality
# -----------------------------
with tabs[8]:
    st.subheader("🧹 Data Quality")
    st.caption("Checks the data before operational decisions are used.")

    dq_summary = data_quality_summary(raw, dq)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", dq_summary["rows"])
    q2.metric("SKUs", dq_summary["skus"])
    q3.metric("Suppliers", dq_summary["suppliers"])
    q4.metric("Warnings", dq_summary["warnings"])
    q5.metric("Critical", dq_summary["critical"])

    if dq_summary["critical"] > 0:
        st.error("⛔ Critical data-quality issues detected.")
    elif dq_summary["warnings"] > 0:
        st.warning("⚠️ Data-quality warnings detected. Review them before issuing purchase decisions.")
    else:
        st.success("✅ All current data-quality checks passed.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Latest period", dq_summary["latest_period"])
        st.metric("Checks completed", dq_summary["checks"])
    with c2:
        st.dataframe(
            dq[["Category","Check","Status","Count","Details"]],
            use_container_width=True, hide_index=True
        )

    st.subheader("📤 Export Data Quality")
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
            st.warning(f"Excel export unavailable: {dq_exc}")
        if dq_excel:
            st.download_button(
                "📗 Data Quality Report (Excel)",
                dq_excel,
                "data_quality_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="data_quality_excel_export"
            )
    st.caption("HTML provides the management-ready visual report; Excel provides editable quality checks and KPI summary.")


# -----------------------------
# Action Plan
# -----------------------------
with tabs[9]:
    st.subheader("📝 Weekly Action Plan")
    st.caption("Planner-ready worklist generated by the Decision Engine.")

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

    st.subheader("Supplier follow-up")
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
    st.markdown("### 📤 Export Action Plan")
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
    st.caption("HTML y Excel utilizan la vista filtrada actual. El HTML incluye KPIs y una presentación ejecutiva; el Excel incluye Summary, Action Plan y Supplier Summary con filtros.")

# -----------------------------
# Change Monitor
# -----------------------------
with tabs[10]:
    st.subheader("🔄 What changed?")
    if not comparison_meta.get("has_comparison"):
        st.info("Necesitas al menos dos periodos históricos para comparar la evolución.")
    else:
        st.caption(f"Comparando **{comparison_meta['previous_period']} → {comparison_meta['current_period']}**")

        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("Purchase need", f"€{comparison_meta['purchase_current']:,.0f}", f"{comparison_meta['purchase_delta']:+,.0f} €")
        cm2.metric("Service risk", f"€{comparison_meta['service_risk_current']:,.0f}", f"{comparison_meta['service_risk_delta']:+,.0f} €")
        cm3.metric("Excess exposure", f"€{comparison_meta['excess_current']:,.0f}", f"{comparison_meta['excess_delta']:+,.0f} €")
        cm4.metric("Critical SKUs", comparison_meta["critical_current"], f"{comparison_meta['critical_delta']:+d}")
        cm5.metric("Action changes", comparison_meta["action_changes"], f"{comparison_meta['worsened']} worsened")

        if comparison_meta["worsened"] > comparison_meta["improved"]:
            st.warning(f"Hay más cambios desfavorables ({comparison_meta['worsened']}) que favorables ({comparison_meta['improved']}).")
        elif comparison_meta["improved"] > comparison_meta["worsened"]:
            st.success(f"La evolución es mayoritariamente favorable: {comparison_meta['improved']} mejorados frente a {comparison_meta['worsened']} empeorados.")
        else:
            st.info("La evolución está equilibrada entre mejoras y empeoramientos.")

        view = comparison[[
            "Priority","SKU","Description","Supplier","Previous_Action","Current_Action",
            "Action_Transition","Previous_Days_Cover","Current_Days_Cover","Days_Cover_Delta",
            "Purchase_Value_Delta","Service_Risk_Delta","Excess_Value_Delta","Sales_Delta_Pct",
            "Change_Classification","Change_Reason"
        ]].copy()
        st.dataframe(view, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top worsened / watch")
            worsened = comparison[comparison["Change_Classification"].isin(["WORSENED","WATCH"])].head(8)
            st.dataframe(
                worsened[["SKU","Description","Current_Action","Days_Cover_Delta",
                          "Purchase_Value_Delta","Service_Risk_Delta","Change_Classification"]],
                use_container_width=True, hide_index=True
            )
        with c2:
            st.subheader("Top improved")
            improved = comparison[comparison["Change_Classification"]=="IMPROVED"].sort_values(
                ["Service_Risk_Delta","Purchase_Value_Delta"], ascending=[True,True]
            ).head(8)
            st.dataframe(
                improved[["SKU","Description","Previous_Action","Current_Action",
                          "Days_Cover_Delta","Purchase_Value_Delta","Service_Risk_Delta"]],
                use_container_width=True, hide_index=True
            )

        change_report_html = build_change_monitor_html(comparison, comparison_meta)

        st.markdown("### 📤 Export Change Monitor")
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
            st.warning(f"Excel export unavailable: {export_exc}")

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
with tabs[11]:
    st.subheader("⚡ Quick analyses")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    quick_question = None
    if q1.button("🛒 Purchase priorities"):
        quick_question = "¿Qué debería comprar esta semana y cuáles son las 3 prioridades más importantes?"
    if q2.button("💰 Reduce inventory"):
        quick_question = "¿Dónde puedo reducir inventario sin aumentar demasiado el riesgo de servicio?"
    if q3.button("🚚 Supplier risk"):
        quick_question = "¿Qué proveedores requieren más atención y por qué?"
    if q4.button("📈 Demand outlook"):
        quick_question = "¿Qué cambios de demanda pueden cambiar mis decisiones de compra?"
    if q5.button("🔄 What changed?"):
        quick_question = "¿Qué ha cambiado entre el último periodo y el anterior y cuáles son las 3 mayores variaciones?"
    if q6.button("📊 Logistics KPIs"):
        quick_question = "¿Cuál es el estado de los principales KPI logísticos y cuáles requieren atención?"
    if quick_question:
        st.session_state.chat.append({"role": "user", "content": quick_question})
        with st.chat_message("user"):
            st.markdown(quick_question)
        with st.chat_message("assistant"):
            with st.spinner("Ejecutando análisis de Supply Chain..."):
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
with tabs[12]:
    st.subheader("📤 Reporting Center")
    st.caption("Visual HTML reports and professional Excel workbooks containing the same decision-ready information.")

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

    st.markdown("### 2. Detailed visual reports")
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
            st.warning(f"Excel export unavailable: {planning_export_exc}")
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
        st.warning(f"Excel export unavailable: {excel_error}")

    st.info("Raw CSV exports remain removed from the reporting workflow. HTML and Excel are now the primary shareable outputs.")


st.divider()
st.caption("Supply Chain AI Copilot V2.0.5 — recommendations require planner validation before execution.")
