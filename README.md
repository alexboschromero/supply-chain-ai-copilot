# Supply Chain AI Copilot V1.1

V1.1 fixes the Copilot failure mode so an invalid/unavailable OpenAI key no longer crashes the app.

## New

- Safe OpenAI key loading from Streamlit Secrets/environment.
- "Test OpenAI connection" button.
- Friendly handling for authentication, rate-limit, API and unexpected errors.
- Automatic local fallback when no API key is configured.
- `secrets.example.toml` template.

## Streamlit Secrets

In Streamlit Community Cloud, use:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

Do not commit the real key to GitHub.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
