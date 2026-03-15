# ============================================================================
# Render.com Dockerfile — Combined API + Dashboard
# ============================================================================
# Context: repo root. Render auto-detects this file.
# Combines FastAPI (API) + Streamlit (Dashboard) behind supervisor + nginx.
# ============================================================================
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ supervisor nginx && rm -rf /var/lib/apt/lists/*

# Python deps — tensorflow-cpu saves ~1GB RAM on free tier
COPY cloud_system/deploy/render/requirements.render.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project into cloud_system/ subdir so imports resolve
COPY cloud_system/ ./cloud_system/

# Create directories
RUN mkdir -p cloud_system/saved_models cloud_system/logs cloud_system/uploads

# Copy supervisor & nginx config
COPY cloud_system/deploy/render/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY cloud_system/deploy/render/nginx.conf /etc/nginx/sites-enabled/default

# Remove default nginx site that conflicts
RUN rm -f /etc/nginx/sites-enabled/default.bak 2>/dev/null; \
    rm -f /etc/nginx/conf.d/default.conf 2>/dev/null; \
    true

# Expose port (Render uses 10000 by default)
EXPOSE 10000

# Start supervisor (manages nginx + api + dashboard)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
