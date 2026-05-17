#!/bin/sh
set -eu

mkdir -p .streamlit

cat > .streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${RECIPEBOX_AUTH_REDIRECT_URI}"
cookie_secret = "${RECIPEBOX_AUTH_COOKIE_SECRET}"

[auth.google]
client_id = "${RECIPEBOX_GOOGLE_CLIENT_ID}"
client_secret = "${RECIPEBOX_GOOGLE_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
