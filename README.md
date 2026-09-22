# Supply Chain AI Copilot V1.0

A cloud-ready Streamlit application for demand, inventory and purchasing decisions.

## Features

- Upload historical CSV
- Inventory risk classification
- Purchase recommendations with MOQ rounding
- Safety-stock calculation using demand variability
- Forecast for next month
- ABC/XYZ segmentation
- Supplier exposure
- Decision Center with prioritized actions
- Scenario simulator
- AI Copilot using OpenAI
- CSV purchase-plan export
- Weekly decision brief export
- Works without an OpenAI API key via local rule-based answers

## Required CSV

`SKU, Description, Supplier, Year, Month, Sales, Stock, Open_PO, Lead_Time_Days, MOQ, Unit_Cost`

A ready-to-test dataset is included at:

`data/sample_history.csv`

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Publish a public URL

The app is structured as a standard Streamlit Community Cloud application. Streamlit's current deployment flow uses a GitHub repository plus an entrypoint file; after deployment it assigns a public `streamlit.app` URL, and a custom subdomain can be selected during deployment.

You must connect your own GitHub/Streamlit accounts to actually publish it. This project cannot create that external account or authorize deployment on your behalf.

## Secrets

For AI mode, add this secret in Streamlit Community Cloud:

```toml
OPENAI_API_KEY = "your-key"
```

Do not commit API keys into Git.

## Recommended production upgrades

- Supabase/Postgres persistence
- Organization/user authentication
- Scheduled data refresh
- ERP/SAP connectors
- Email alerts
- Stripe billing
- Forecast model selection
- Demand anomaly detection
- Approval workflows for purchase recommendations
- Monitoring, backups and GDPR controls
