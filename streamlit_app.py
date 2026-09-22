
import io, os, math
from datetime import date, timedelta
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

def data_quality_report(df):
    rows = []
    for col in REQUIRED:
        if col not in df.columns:
            rows.append({"Field": col, "Status": "MISSING", "Issue": "Required field not found"})
            continue
        nulls = int(df[col].isna().sum())
        rows.append({
            "Field": col,
            "Status": "OK" if nulls == 0 else "WARNING",
            "Issue": "No missing values" if nulls == 0 else f"{nulls} missing values"
        })

    dupes = int(df.duplicated(subset=["SKU","Year","Month"]).sum()) if all(
        c in df.columns for c in ["SKU","Year","Month"]
    ) else 0
    rows.append({
        "Field": "SKU-Year-Month",
        "Status": "OK" if dupes == 0 else "WARNING",
        "Issue": "No duplicate periods" if dupes == 0 else f"{dupes} duplicate rows"
    })
    return pd.DataFrame(rows)

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

def route_agent(question, a):
    q = question.lower()
    tools = []
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
        tools = [
            agent_purchase_tool(a), agent_service_tool(a), agent_inventory_tool(a),
            agent_supplier_tool(a), agent_forecast_tool(a)
        ]
    seen = set()
    out = []
    for t in tools:
        if t["name"] not in seen:
            out.append(t)
            seen.add(t["name"])
    return out

def agent_local_response(question, a):
    tools = route_agent(question, a)
    lines = ["## Supply Chain Agent — análisis"]
    for tool in tools:
        lines.append(f"### {tool['name']}")
        if tool["name"] == "purchase_planner":
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

def ai_chat(question, a, api_key=None, model="gpt-5.6-luna"):
    tool_results = route_agent(question, a)

    if not api_key or OpenAI is None:
        return agent_local_response(question, a)

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
st.caption("From raw supply-chain data to prioritized decisions · V1.5.1")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SKUs", K["sku"])
c2.metric("🔴 Critical", K["critical"])
c3.metric("🟠 Review", K["review"])
c4.metric("🛒 Purchase need", f"€{K['purchase']:,.0f}")
c5.metric("💰 Inventory", f"€{K['inventory']:,.0f}")
c6.metric("📈 Next month", f"{K['forecast']:,.0f}")

tabs = st.tabs([
    "🎯 Decision Center","📊 Inventory","📈 Forecast",
    "🧩 ABC/XYZ","🚚 Suppliers","🧪 Scenarios","🧹 Data Quality",
    "📝 Action Plan","🤖 Copilot","📤 Export"
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

# -----------------------------
# Inventory
# -----------------------------
with tabs[1]:
    st.subheader("Inventory health")
    left, right = st.columns(2)
    with left:
        status_counts = a["Status"].value_counts()
        st.bar_chart(status_counts)
    with right:
        st.bar_chart(a.groupby("Supplier")["Inventory_Value"].sum())
    st.dataframe(
        a[[
            "SKU","Description","Supplier","Status","Stock","Open_PO",
            "Days_Cover","Safety_Stock","Recommended_Order",
            "Inventory_Value","ABC_XYZ","Action","Action_Timing","Decision_Confidence"
        ]],
        use_container_width=True, hide_index=True
    )

# -----------------------------
# Forecast
# -----------------------------
with tabs[2]:
    st.subheader("Demand outlook")
    f = a[[
        "SKU","Description","Annual_Sales","Avg_Monthly_Demand",
        "Forecast_Next_Month","Forecast_Change_Pct",
        "Trend_Units_Per_Month","Demand_CV"
    ]].sort_values("Forecast_Next_Month", ascending=False)
    st.dataframe(f, use_container_width=True, hide_index=True)
    st.info(
        "El forecast del MVP utiliza una media ponderada de los últimos 6 meses más una tendencia lineal. "
        "La siguiente iteración puede añadir estacionalidad, demanda intermitente y modelos alternativos."
    )

# -----------------------------
# ABC/XYZ
# -----------------------------
with tabs[3]:
    st.subheader("Segmentation")
    st.dataframe(
        a[[
            "SKU","Description","Annual_Consumption_Value",
            "ABC","Demand_CV","XYZ","ABC_XYZ"
        ]].sort_values("Annual_Consumption_Value", ascending=False),
        use_container_width=True, hide_index=True
    )

# -----------------------------
# Suppliers
# -----------------------------
with tabs[4]:
    st.subheader("Supplier exposure")
    sup = a.groupby("Supplier", as_index=False).agg(
        SKUs=("SKU","count"),
        Inventory_Value=("Inventory_Value","sum"),
        Purchase_Value=("Purchase_Value","sum"),
        Critical=("Status", lambda s: (s=="🔴 CRITICAL").sum()),
        Avg_Cover=("Days_Cover","mean")
    ).sort_values(["Critical","Inventory_Value"], ascending=[False,False])
    st.dataframe(sup, use_container_width=True, hide_index=True)

# -----------------------------
# Scenarios
# -----------------------------
with tabs[5]:
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


# -----------------------------
# Data Quality
# -----------------------------
with tabs[8]:
    st.subheader("🧹 Data quality")
    st.caption("The Copilot should not make decisions from bad master or transaction data.")
    dq = data_quality_report(raw)
    st.dataframe(dq, use_container_width=True, hide_index=True)

    issues = dq[dq["Status"] != "OK"]
    if issues.empty:
        st.success("✅ No structural data-quality issues detected.")
    else:
        st.warning(f"Detected {len(issues)} data-quality warning(s). Resolve them before using recommendations operationally.")

# -----------------------------
# Action Plan
# -----------------------------
with tabs[9]:
    st.subheader("📝 Weekly action plan")
    st.caption("Operational worklist created from the Decision Engine.")
    plan = build_action_plan(a)
    st.dataframe(
        plan[[
            "Priority","SKU","Description","Supplier","Action",
            "Owner","Deadline","Reason","Confidence","Purchase_Value"
        ]],
        use_container_width=True,
        hide_index=True
    )

    selected_sku = st.selectbox(
        "Generate supplier communication for SKU",
        options=[""] + list(a["SKU"].astype(str))
    )
    if selected_sku:
        row = a[a["SKU"].astype(str) == selected_sku].iloc[0]
        st.code(supplier_message(row), language="text")

        st.download_button(
            "⬇️ Download supplier message",
            supplier_message(row).encode("utf-8"),
            f"supplier_message_{selected_sku}.txt",
            "text/plain"
        )

    st.download_button(
        "⬇️ Download action plan",
        plan.to_csv(index=False).encode("utf-8"),
        "weekly_action_plan.csv",
        "text/csv"
    )

# -----------------------------
# Copilot
# -----------------------------
with tabs[8]:
    st.subheader("⚡ Quick analyses")
    q1, q2, q3, q4 = st.columns(4)
    quick_question = None
    if q1.button("🛒 Purchase priorities"):
        quick_question = "¿Qué debería comprar esta semana y cuáles son las 3 prioridades más importantes?"
    if q2.button("💰 Reduce inventory"):
        quick_question = "¿Dónde puedo reducir inventario sin aumentar demasiado el riesgo de servicio?"
    if q3.button("🚚 Supplier risk"):
        quick_question = "¿Qué proveedores requieren más atención y por qué?"
    if q4.button("📈 Demand outlook"):
        quick_question = "¿Qué cambios de demanda pueden cambiar mis decisiones de compra?"
    if quick_question:
        st.session_state.chat.append({"role": "user", "content": quick_question})
        with st.chat_message("user"):
            st.markdown(quick_question)
        with st.chat_message("assistant"):
            with st.spinner("Ejecutando análisis de Supply Chain..."):
                ans = ai_chat(quick_question, a, effective_key, model)
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
                ans = ai_chat(q, a, effective_key, model)
            st.markdown(ans)
        st.session_state.chat.append({"role":"assistant","content":ans})

# -----------------------------
# Export
# -----------------------------
with tabs[9]:
    st.subheader("📤 Export")
    st.download_button(
        "Download full analysis",
        a.to_csv(index=False).encode("utf-8"),
        "supply_chain_analysis.csv",
        "text/csv",
        use_container_width=True
    )
    po = export_purchase(a)
    if not po.empty:
        st.download_button(
            "Download purchase plan",
            po.to_csv(index=False).encode("utf-8"),
            "purchase_plan.csv",
            "text/csv",
            use_container_width=True
        )

    report = decision_text(a)
    st.download_button(
        "Download weekly decision brief",
        report.encode("utf-8"),
        "weekly_decision_brief.md",
        "text/markdown",
        use_container_width=True
    )

st.divider()
st.caption("Supply Chain AI Copilot V1.0 — recommendations require planner validation before execution.")
