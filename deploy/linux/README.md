# Migrating LiNT-II to the Strato Linux box

This moves the demo from the two-box setup (home Mac Studio running the app,
dialing an SSH reverse tunnel into the Strato front) to a **single box**: the
existing Strato front (85.215.105.128) runs the app directly under systemd,
nginx proxies to it on loopback, and the tunnel is retired.

Nothing about the LLM changes — it stays on the **Mistral API**
(`LINT_PROVIDER=mistral`). MLX is Apple-only and is not used here.

## Target architecture

```
client --443--> nginx (this box) --proxy--> 127.0.0.1:8000 uvicorn (this box, systemd)
                                                            |
                                                    Mistral API (HTTPS out)
```

The proxy target (`127.0.0.1:8000`) is **the same address the SSH tunnel used to
publish**, so `proxy_pass` does not change — only the thing listening on that
port does (local uvicorn instead of the tunnel).

DNS is unchanged: `lint-ii.valkuil.net A -> 85.215.105.128`.

## Why one uvicorn process

The job store (`_jobs`) and the thread pools live **in process**. Running uvicorn
with `--workers N` would give each worker its own job dict, so an
`/analyze-result` poll could land on a worker that never ran the job. Keep a
single process; concurrency comes from the in-process pools
(`LINT_II_ANALYSIS_WORKERS`, `LINT_II_GENERAL_WORKERS`).

## Procedure

Run these on the Strato box (as root / with sudo). A Claude Code instance on the
box can execute them.

### 1. Provision (does not touch the live tunnel/nginx yet)

```bash
sudo bash /opt/lint_ii/deploy/linux/setup.sh   # clones to /opt/lint_ii if absent
```

The script installs system deps (python3.11, pandoc, poppler-utils, nginx,
certbot), a `lint` service user, a venv with `pip install -e ".[llm,server]"`,
the spaCy model `nl_core_news_lg`, `/var/log/lint-ii`, and the systemd unit.

### 2. Configure secrets

```bash
sudoedit /etc/lint-ii/lint-ii.env      # set MISTRAL_API_KEY (the rest is filled in)
```

The demo's key is the same one currently in the Mac's LaunchAgents plist. The
env file is `chmod 640 root:lint` and is **not** in the repo.

### 3. Preflight (does NOT bind port 8000 — the tunnel still holds it)

The SSH reverse tunnel from the Mac occupies `127.0.0.1:8000` on the box, so
don't start the service yet. Instead validate the install without binding:

```bash
# Deps, imports, and paths load cleanly:
sudo -u lint HOME=/opt/lint_ii /opt/lint_ii/.venv/bin/python -c "import api; print('import OK')"
# spaCy Dutch model is present and loadable:
sudo -u lint /opt/lint_ii/.venv/bin/python -c "import spacy; spacy.load('nl_core_news_lg'); print('spaCy OK')"
```

Both printing OK means the box is ready; the only untested piece is the live
Mistral call, which the first real analysis exercises after cutover.

### 4. Cutover (the only step with a brief outage)

Run back to back to minimize the window:

1. **On the Mac** — stop the tunnel and app so port 8000 frees up on the box:
   ```bash
   launchctl unload ~/Library/LaunchAgents/net.valkuil.lint-ii-tunnel.plist
   launchctl unload ~/Library/LaunchAgents/net.valkuil.lint-ii.plist
   ```
2. **On the box** — start the service (now it can bind 8000), then nginx:
   ```bash
   sudo systemctl enable --now net.valkuil.lint-ii
   sleep 5
   curl -s http://127.0.0.1:8000/health     # {"status":"ok","model":"mistral-large-latest",...}
   sudo nginx -t && sudo systemctl reload nginx
   ```
3. **From anywhere** — verify the public endpoint and that the fixes hold:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" https://lint-ii.valkuil.net/            # 200
   curl -s -o /dev/null -w "%{http_code}\n" https://lint-ii.valkuil.net/.git/config # 404
   curl -s -o /dev/null -w "%{http_code}\n" https://lint-ii.valkuil.net/api.py      # 404
   ```
   Then load the demo in a browser and run one analysis to exercise the live
   Mistral path end to end.

### 5. nginx config

The hardened config is already in the repo:
`deploy/strato-nginx-lint-ii.conf` + `deploy/strato-nginx-lint-ii-proxy.conf`
(rate limits, per-IP connection cap, slow-client timeouts, dotfile 404). Install:

```bash
sudo cp /opt/lint_ii/deploy/strato-nginx-lint-ii.conf       /etc/nginx/sites-available/lint-ii
sudo cp /opt/lint_ii/deploy/strato-nginx-lint-ii-proxy.conf /etc/nginx/snippets/lint-ii-proxy.conf
sudo ln -sf /etc/nginx/sites-available/lint-ii /etc/nginx/sites-enabled/lint-ii
sudo nginx -t && sudo systemctl reload nginx
```

Its `proxy_pass` is unchanged from the tunnel era; the header comment still
describes the tunnel and can be trimmed after cutover (cosmetic only).

## Operating the service

```bash
sudo systemctl status net.valkuil.lint-ii
sudo journalctl -u net.valkuil.lint-ii -f      # stderr (INFO+)
tail -f /var/log/lint-ii/app.log               # app log (INFO)
sudo systemctl restart net.valkuil.lint-ii     # after a git pull
```

Code update: `sudo -u lint git -C /opt/lint_ii pull --ff-only && sudo systemctl restart net.valkuil.lint-ii`
(or re-run `setup.sh`, which also refreshes deps).

## Rollback

If anything is wrong, bring the Mac back:

```bash
# On the box:
sudo systemctl stop net.valkuil.lint-ii
# On the Mac:
launchctl load ~/Library/LaunchAgents/net.valkuil.lint-ii.plist
launchctl load ~/Library/LaunchAgents/net.valkuil.lint-ii-tunnel.plist
```

nginx needs no change to roll back — it still proxies to `127.0.0.1:8000`, which
the tunnel republishes. Keep the Mac units in place until the box has run a full
tester day cleanly.

## What carries over automatically

All the security fixes are in the code/config the box pulls: the whitelisted
static mount (no repo-root exposure), the `/analyze` flood guard (429), the
INFO-default logging, and the nginx rate/connection limits. No extra steps.
