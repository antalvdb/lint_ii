#!/usr/bin/env bash
# Provision the LiNT-II demo on a Debian/Ubuntu box. Run as root (or with sudo).
# Safe to re-run: it pulls the latest code and reinstalls deps.
#
#   sudo bash deploy/linux/setup.sh
#
# Afterwards: put the Mistral key in /etc/lint-ii/lint-ii.env, then
#   systemctl enable --now net.valkuil.lint-ii
set -euo pipefail

# Overridable for an existing checkout, e.g.:
#   sudo APP_DIR=/home/antalb/servers/lint_ii SVC_USER=antalb bash deploy/linux/setup.sh
# When SVC_USER is a normal login account the system-user creation is skipped and
# ownership/HOME follow that account. If you point at an existing checkout, also
# edit net.valkuil.lint-ii.service (User, HOME, WorkingDirectory, ExecStart path)
# to match before installing it.
APP_DIR="${APP_DIR:-/opt/lint_ii}"
REPO="${REPO:-https://github.com/antalvdb/lint_ii.git}"
SVC_USER="${SVC_USER:-lint}"
PY="${PY:-python3.11}"
SVC_HOME="$(getent passwd "$SVC_USER" 2>/dev/null | cut -d: -f6)"
SVC_HOME="${SVC_HOME:-$APP_DIR}"

echo "== 1. System packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update
# python3.11 is default on Debian 12 / Ubuntu 24.04; fall back to python3 if the
# versioned package is unavailable.
apt-get install -y "$PY" "${PY}-venv" 2>/dev/null || { PY=python3; apt-get install -y python3 python3-venv; }
apt-get install -y python3-pip git pandoc poppler-utils nginx certbot python3-certbot-nginx

echo "== 2. Service user =="
id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$SVC_USER"

echo "== 3. Code =="
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO" "$APP_DIR"
fi

echo "== 4. Virtualenv + Python deps =="
[ -d "$APP_DIR/.venv" ] || "$PY" -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -e "${APP_DIR}[llm,server]"
echo "== 4b. spaCy Dutch model (nl_core_news_lg, ~560 MB) =="
"$APP_DIR/.venv/bin/python" -m spacy download nl_core_news_lg

echo "== 5. Directories, env file, ownership =="
mkdir -p /var/log/lint-ii /etc/lint-ii
if [ ! -f /etc/lint-ii/lint-ii.env ]; then
    cp "$APP_DIR/deploy/linux/lint-ii.env.example" /etc/lint-ii/lint-ii.env
    echo "   -> created /etc/lint-ii/lint-ii.env from the template; EDIT IT to add MISTRAL_API_KEY."
fi
chown -R "$SVC_USER:$SVC_USER" "$APP_DIR" /var/log/lint-ii
chown root:"$SVC_USER" /etc/lint-ii/lint-ii.env
chmod 640 /etc/lint-ii/lint-ii.env

echo "== 6. systemd unit =="
cp "$APP_DIR/deploy/linux/net.valkuil.lint-ii.service" /etc/systemd/system/
systemctl daemon-reload

echo
echo "Done. Next:"
echo "  1. Edit /etc/lint-ii/lint-ii.env and set MISTRAL_API_KEY."
echo "  2. systemctl enable --now net.valkuil.lint-ii"
echo "  3. curl -s http://127.0.0.1:8000/health   # expect {\"status\":\"ok\",...}"
echo "  4. Install the nginx config (see deploy/linux/README.md), nginx -t && systemctl reload nginx."
