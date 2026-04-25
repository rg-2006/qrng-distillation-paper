#!/bin/bash
#
# CloudFlare Tunnel setup
# =======================
#
# Exposes your gRPC server externally without configuring inbound ports
# on the Lambda Labs droplet. Useful for letting a remote client connect.
#
# Two modes are supported:
#
#   1. QUICK MODE (anonymous, ephemeral URL — no Cloudflare account needed)
#      Just runs `cloudflared tunnel --url ...` and prints a *.trycloudflare.com URL.
#      Good for benchmarks and demos. URL changes every restart.
#
#   2. NAMED TUNNEL (requires free Cloudflare account + domain)
#      Persistent URL backed by your domain. Recommended for sustained use.
#      Requires you to authenticate with `cloudflared tunnel login` first.
#
# Usage:
#   bash scripts/setup_cloudflare.sh quick      # ephemeral, no account
#   bash scripts/setup_cloudflare.sh named NAME # persistent, requires account
#
# The gRPC server should be running on localhost:50051 before you start
# the tunnel (in another terminal:  python -m src.server.grpc_server).
# ----------------------------------------------------------------------

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()    { echo -e "${GREEN}[cloudflare]${NC} $1"; }
warn()   { echo -e "${YELLOW}[warn]${NC}       $1"; }
fatal()  { echo -e "${RED}[fatal]${NC}      $1"; exit 1; }

cd "$(dirname "$0")/.."

# ----------------------------------------------------------------------
# 1. Install cloudflared if needed
# ----------------------------------------------------------------------

if ! command -v cloudflared &> /dev/null; then
    log "Installing cloudflared..."
    ARCH=$(dpkg --print-architecture)
    if [ "$ARCH" = "amd64" ]; then
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    elif [ "$ARCH" = "arm64" ]; then
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
    else
        fatal "Unsupported architecture: $ARCH"
    fi
    curl -L -o /tmp/cloudflared.deb "$URL"
    sudo dpkg -i /tmp/cloudflared.deb || sudo apt-get install -f -y
    rm /tmp/cloudflared.deb
fi

log "cloudflared version: $(cloudflared --version | head -n1)"

# ----------------------------------------------------------------------
# 2. Mode dispatch
# ----------------------------------------------------------------------

MODE="${1:-quick}"
LOCAL_PORT=50051

case "$MODE" in
    quick)
        log "============================================================"
        log "QUICK MODE — ephemeral *.trycloudflare.com tunnel"
        log "============================================================"
        log ""
        log "This will start a tunnel that exposes localhost:${LOCAL_PORT}"
        log "to a public *.trycloudflare.com URL. Note the URL printed below"
        log "and use it as the --server argument when running the client."
        log ""
        log "NOTE: The tunnel uses HTTP/2 which gRPC requires. Cloudflare"
        log "automatically negotiates this for gRPC traffic. The URL will"
        log "be HTTPS-prefixed; configure your gRPC client to use TLS."
        log ""
        log "Press Ctrl+C to stop the tunnel."
        log ""
        cloudflared tunnel --url "http://localhost:${LOCAL_PORT}" --no-autoupdate
        ;;

    named)
        TUNNEL_NAME="${2:-qrng-entropy}"
        log "============================================================"
        log "NAMED TUNNEL MODE: $TUNNEL_NAME"
        log "============================================================"

        # Check authentication
        if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
            log ""
            log "You need to authenticate with Cloudflare first."
            log "A browser window will open; log in to your Cloudflare account."
            log ""
            cloudflared tunnel login
        fi

        # Create tunnel if it doesn't exist
        if ! cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
            log "Creating tunnel: $TUNNEL_NAME"
            cloudflared tunnel create "$TUNNEL_NAME"
        else
            log "Tunnel $TUNNEL_NAME already exists."
        fi

        TUNNEL_ID=$(cloudflared tunnel list 2>/dev/null | grep "$TUNNEL_NAME" | awk '{print $1}')
        log "Tunnel ID: $TUNNEL_ID"

        # Write a config
        CONFIG_FILE="$HOME/.cloudflared/config.yml"
        log "Writing tunnel config to $CONFIG_FILE"
        cat > "$CONFIG_FILE" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $HOME/.cloudflared/${TUNNEL_ID}.json

ingress:
  - service: http://localhost:${LOCAL_PORT}
EOF

        log ""
        log "Now configure a DNS route. Run (replace yourdomain.com):"
        log "  cloudflared tunnel route dns $TUNNEL_NAME entropy.yourdomain.com"
        log ""
        log "Then start the tunnel:"
        log "  cloudflared tunnel run $TUNNEL_NAME"
        log ""
        log "Once running, your gRPC server is reachable at:"
        log "  https://entropy.yourdomain.com"
        ;;

    *)
        fatal "Unknown mode: $MODE.  Use 'quick' or 'named <name>'."
        ;;
esac
