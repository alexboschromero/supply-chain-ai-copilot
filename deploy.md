# Deployment checklist

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload this project.
3. Open Streamlit Community Cloud.
4. Connect GitHub.
5. Create an app using `streamlit_app.py` as the entrypoint.
6. In Advanced settings, add `OPENAI_API_KEY`.
7. Deploy.
8. Set the generated/custom `streamlit.app` subdomain.

## Important

For a commercial SaaS, use cloud Postgres/Supabase instead of local files for persistent customer data. Do not rely on local Streamlit disk storage for multi-user production data.
