# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Link previews (WhatsApp, Slack, LinkedIn, Google) read the raw HTML and
# don't run JavaScript, so they see Streamlit's generic
# "<title>Streamlit</title>" instead of our page_title. Patch the static
# page with a real title and preview tags. Fails the build if Streamlit
# ever changes that markup, rather than silently showing "Streamlit".
RUN python - <<'EOF'
import html, pathlib, streamlit

page = pathlib.Path(streamlit.__file__).parent / "static" / "index.html"
source = page.read_text(encoding="utf-8")
marker = "<title>Streamlit</title>"
assert marker in source, f"{marker} not found in {page}"

title = "AI Customer Support"
description = (
    "Multi-agent AI customer support: built-in knowledge, live web search "
    "and a reconciliation agent combined into one clear answer."
)
t, d = html.escape(title), html.escape(description)

head = f"""<title>{t}</title>
    <meta name="description" content="{d}" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="supportdude.sbs" />
    <meta property="og:title" content="{t}" />
    <meta property="og:description" content="{d}" />
    <meta name="twitter:card" content="summary" />
    <meta name="twitter:title" content="{t}" />
    <meta name="twitter:description" content="{d}" />"""

page.write_text(source.replace(marker, head, 1), encoding="utf-8")
print(f"Patched {page}")
EOF

COPY app.py .
COPY .streamlit/config.toml .streamlit/config.toml

# Run as a non-root user; it owns /app so save_to_file can write answers.txt
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
