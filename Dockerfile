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

# Streamlit listens on HF's default Docker port 7860 (declared as app_port in
# README) so HF's health check reaches it with no port ambiguity.
EXPOSE 7860

# TEMPORARY DIAGNOSTIC LAUNCH: echo checkpoints before Streamlit so the HF
# Container log shows exactly how far the container gets. Revert to the plain
# `streamlit run ...` CMD once we've identified the failure.
CMD ["sh", "-c", "echo '>>> [1] container CMD is running'; python --version; echo '>>> [2] importing streamlit + pandas...'; python -c 'import streamlit, pandas; print(\">>> [3] imports OK, streamlit\", streamlit.__version__)'; echo '>>> [4] launching streamlit on 7860'; exec streamlit run ui/streamlit_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --logger.level=info"]
