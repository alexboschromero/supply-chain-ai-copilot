
import io, os, math
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

    def status(r):
        if r["Stock"] < r["Lead_Time_Demand"]:
            return "🔴 CRITICAL"
        if r["Stock"] < r["Required_Stock"]:
            return "🟠 REVIEW"
        if r["Days_Cover"] > r["Lead_Time_Days"] + safety_days*3:
            return "🟡 EXCESS"
        return "🟢 OK"
    a["Status"] = a.apply(status, axis=1)

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
    lines = ["### Prioridades de esta semana"]
    for _, r in top.iterrows():
        if r["Status"] == "🔴 CRITICAL":
            action = f"Comprar {r['Recommended_Order']:.0f} uds."
        elif r["Status"] == "🟠 REVIEW":
            action = f"Revisar reposición; sugerencia {r['Recommended_Order']:.0f} uds."
        elif r["Status"] == "🟡 EXCESS":
            action = "Revisar exceso / aplazar compra."
        else:
            action = "Mantener política."
        lines.append(
            f"- **{r['SKU']} — {r['Description']}**: {action} "
            f"(cobertura {r['Days_Cover']:.1f} días; proveedor {r['Supplier']})."
        )
    return "\n".join(lines)

def build_context(a):
    cols = [
        "SKU","Description","Supplier","Status","Stock","Open_PO",
        "Days_Cover","Lead_Time_Days","Recommended_Order","Purchase_Value",
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
    q = question.lower()

    def local_mode():
        if any(w in q for w in ["compr", "buy", "orden", "purchase"]):
            return decision_text(a)

        if any(w in q for w in ["rotura", "riesgo", "stockout", "critical"]):
            x = a[a["Status"] == "🔴 CRITICAL"]
            if x.empty:
                return "No se detectan riesgos críticos con los parámetros actuales."

            return "### Riesgos críticos\n" + "\n".join(
                f"- **{r.SKU}**: {r.Days_Cover:.1f} días de cobertura frente a "
                f"{r.Lead_Time_Days:.0f} días de lead time."
                for _, r in x.sort_values("Decision_Score", ascending=False).iterrows()
            )

        if any(w in q for w in ["exceso", "sobrestock", "excess"]):
            x = a[a["Status"] == "🟡 EXCESS"]
            if x.empty:
                return "No se detectan excesos."

            return "### Exceso de cobertura\n" + "\n".join(
                f"- **{r.SKU}**: {r.Days_Cover:.1f} días de cobertura."
                for _, r in x.iterrows()
            )

        if "forecast" in q or "previsión" in q:
            x = a.sort_values("Forecast_Next_Month", ascending=False).head(10)
            return "### Forecast próximo mes\n" + "\n".join(
                f"- **{r.SKU}**: {r.Forecast_Next_Month:.0f} uds. "
                f"({r.Forecast_Change_Pct:+.1f}% vs media)."
                for _, r in x.iterrows()
            )

        if "proveedor" in q or "supplier" in q:
            x = a.groupby("Supplier", as_index=False).agg(
                Critical=("Status", lambda s: (s == "🔴 CRITICAL").sum()),
                Inventory_Value=("Inventory_Value", "sum"),
                Purchase_Value=("Purchase_Value", "sum")
            ).sort_values(["Critical", "Inventory_Value"], ascending=[False, False])

            return "### Proveedores a vigilar\n" + "\n".join(
                f"- **{r.Supplier}**: {int(r.Critical)} SKUs críticos; "
                f"inventario €{r.Inventory_Value:,.0f}; compras €{r.Purchase_Value:,.0f}."
                for _, r in x.head(10).iterrows()
            )

        return (
            "Puedo ayudarte con compras, riesgos de rotura, exceso, "
            "forecast y proveedores."
        )

    if not api_key:
        return local_mode()

    if OpenAI is None:
        return "OpenAI no está instalado. Usa el modo local."

    try:
        client = _make_openai_client(api_key)

        context = build_context(a)

        response = client.responses.create(
            model=model,
            instructions=(
                "Eres el Supply Chain AI Copilot de una empresa. Responde en español. "
                "Usa exclusivamente los datos proporcionados. No inventes números. "
                "Primero resume los hechos y luego recomienda acciones concretas. "
                "Prioriza riesgo de stockout, impacto económico y nivel de servicio. "
                "Si faltan datos, dilo. No ejecutes compras: prepara recomendaciones "
                "para validación humana."
            ),
            input=f"PREGUNTA:\n{question}\n\nDATOS CALCULADOS:\n{context}",
        )

        return response.output_text

    except AuthenticationError as exc:
        status, code, error_type = _openai_error_info(exc)
        return (
            "### ⚠️ OpenAI ha rechazado la autenticación\n\n"
            f"HTTP: `{status or 'desconocido'}` · "
            f"code: `{code or 'desconocido'}` · "
            f"type: `{error_type or 'desconocido'}`\n\n"
            "Revisa la API key y el proyecto configurados en Streamlit Secrets."
        )

    except RateLimitError as exc:
        status, code, error_type = _openai_error_info(exc)
        return (
            "### ⚠️ Límite o cuota de OpenAI\n\n"
            f"HTTP: `{status or 'desconocido'}` · "
            f"code: `{code or 'desconocido'}` · "
            f"type: `{error_type or 'desconocido'}`"
        )

    except APIError as exc:
        status, code, error_type = _openai_error_info(exc)
        return (
            "### ⚠️ Error de API OpenAI\n\n"
            f"HTTP: `{status or 'desconocido'}` · "
            f"code: `{code or 'desconocido'}` · "
            f"type: `{error_type or 'desconocido'}`"
        )

    except Exception as exc:
        status, code, error_type = _openai_error_info(exc)
        return (
            "### ⚠️ Error al consultar OpenAI\n\n"
            f"`{type(exc).__name__}` · "
            f"HTTP `{status or 'desconocido'}` · "
            f"code `{code or 'desconocido'}` · "
            f"type `{error_type or 'desconocido'}`"
        )

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
    uploaded = st.file_uploader("CSV histórico", type=["csv"])
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
        st.session_state.analysis = analyze(sample_data(), safety_days, service)
        st.session_state.chat = []
        st.rerun()

# -----------------------------
# Load data
# -----------------------------
if uploaded:
    raw = pd.read_csv(uploaded)
    valid, error = validate(raw)
    if not valid:
        st.error(error)
        st.stop()
    st.session_state.analysis = analyze(raw, safety_days, service)
elif st.session_state.analysis is None:
    st.session_state.analysis = analyze(sample_data(), safety_days, service)

a = st.session_state.analysis
K = kpis(a)

# Add forecast change vs historical monthly average
a["Forecast_Change_Pct"] = np.where(
    a["Avg_Monthly_Demand"] > 0,
    (a["Forecast_Next_Month"]/a["Avg_Monthly_Demand"]-1)*100,
    0
)

# -----------------------------
# Header
# -----------------------------
st.title("📦 Supply Chain AI Copilot")
st.caption("From raw supply-chain data to prioritized decisions.")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SKUs", K["sku"])
c2.metric("🔴 Critical", K["critical"])
c3.metric("🟠 Review", K["review"])
c4.metric("🛒 Purchase need", f"€{K['purchase']:,.0f}")
c5.metric("💰 Inventory", f"€{K['inventory']:,.0f}")
c6.metric("📈 Next month", f"{K['forecast']:,.0f}")

tabs = st.tabs([
    "🎯 Decision Center","📊 Inventory","📈 Forecast",
    "🧩 ABC/XYZ","🚚 Suppliers","🧪 Scenarios","🤖 Copilot","📤 Export"
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
            "SKU","Description","Supplier","Status","Days_Cover",
            "Lead_Time_Days","Recommended_Order","Purchase_Value","Decision_Score"
        ]],
        use_container_width=True, hide_index=True
    )

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
            "Inventory_Value","ABC_XYZ"
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

    sim = raw.copy() if uploaded else sample_data().copy()
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
# Copilot
# -----------------------------
with tabs[6]:
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
with tabs[7]:
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
