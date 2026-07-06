#!/usr/bin/env bash
# Configure chrony as NTP server for LAN (192.168.1.0/24).
# Usage: bash install_scripts/setup_chrony_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="/etc/chrony/chrony.conf"
SNIPPET="$SCRIPT_DIR/chrony-server-snippet.conf"
MARKER="# >>> SONIC / local NTP server >>>"

if [ ! -f "$CONF" ]; then
    echo "[ERROR] $CONF not found. Install chrony: sudo apt install chrony"
    exit 1
fi

if grep -qF "$MARKER" "$CONF" 2>/dev/null; then
    echo "[OK] NTP server block already present in $CONF"
else
    echo "[INFO] Backing up $CONF ..."
    sudo cp "$CONF" "${CONF}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] Appending server settings ..."
    {
        echo ""
        echo "$MARKER"
        cat "$SNIPPET"
        echo "# <<< SONIC / local NTP server <<<"
    } | sudo tee -a "$CONF" > /dev/null
    echo "[OK] Updated $CONF"
fi

echo "[INFO] Restarting chrony ..."
sudo systemctl restart chrony
sudo systemctl enable chrony

echo ""
echo "=== chrony sources ==="
chronyc sources 2>/dev/null | head -10 || true
echo ""
echo "=== chrony tracking ==="
chronyc tracking 2>/dev/null | head -8 || true
echo ""
echo "LAN clients can sync to: $(hostname -I | awk '{print $1}') (UDP 123)"
echo "On client: sudo apt install chrony && set server in chrony.conf to this IP"
