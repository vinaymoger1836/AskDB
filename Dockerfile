# AskDB on Hugging Face Spaces (Docker SDK).
#
# HF's legacy native Streamlit launcher was hanging at startup ("stuck at
# Starting"). Running Streamlit ourselves via Docker gives an explicit,
# debuggable launch command and full control over the port and flags.

FROM python:3.12-slim

# Unbuffered stdout/stderr so Streamlit's startup banner (or any error) shows up
# immediately in the HF Container log instead of being swallowed by buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Bake the seeded demo database into the image so the app has read-only data at
# runtime and never needs to write to the filesystem to serve a query.
RUN python -m data.seed

# Streamlit listens here; HF proxies this port (declared as app_port in README).
EXPOSE 8501

# Launch Streamlit explicitly. headless=true skips the first-run email prompt;
# address 0.0.0.0 makes it reachable by HF's proxy.
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
