# Deployment: Strato 443 front + SSH reverse tunnel → home Mac Studio

## Problem this solves

The demo runs the (fast) inference on the Mac Studio at home, but many
restrictive client networks — mobile hotspots, corporate/guest WiFi, some
carriers — only allow outbound **80/443**, so testers on those networks see
*"the network did not allow a connection"* when hitting the home server
directly on its non-standard port.

Fix: the Strato Linux box (`85.215.105.128`, a data-center host) terminates TLS
on the standard **443** — reachable on virtually every network — and reverse-
proxies to the home Mac Studio.

## How the home server is reached: an outbound SSH reverse tunnel

The Mac Studio **dials out** to Strato over SSH and publishes its local server
on Strato's loopback. Strato's nginx proxies 443 to that loopback port.

```
client ──443──▶ Strato (85.215.105.128) ──┐
                                           │  nginx proxy_pass
                                           ▼
                               127.0.0.1:8000  on Strato
                                           ▲
                                           │  SSH reverse tunnel (-R), dialed
                                           │  OUTBOUND by the Mac
                               127.0.0.1:8000  on the home Mac Studio (uvicorn)
```

Because the Mac initiates the connection, this removes three fragile pieces the
old direct-to-home design needed:

- **no dynamic DNS / `origin` record** — the home IP can change freely; nothing
  tracks it. The only DNS record is `lint-ii.valkuil.net A → 85.215.105.128`.
- **no router port-forward** — nothing inbound to home is required.
- **no home TLS cert** — the SSH transport encrypts the hop, so the Mac serves
  plain HTTP on loopback. Only Strato holds a (public) cert.

No application code changes are involved: the frontend uses only relative URLs
and the API hardcodes no host/port, so it works unchanged behind the proxy.

## Components

| Where | What | File |
|---|---|---|
| Strato | nginx 443 → `127.0.0.1:8000` | `strato-nginx-lint-ii.conf` |
| Mac | uvicorn on `127.0.0.1:8000` (launchd) | `mac/net.valkuil.lint-ii.plist` |
| Mac | SSH reverse tunnel to Strato (launchd) | `mac/net.valkuil.lint-ii-tunnel.plist` |

Both Mac jobs are launchd agents with `KeepAlive` + `RunAtLoad`, so each
restarts if it dies, and both start automatically when the `antalb` user's
desktop session begins. This replaces the old manual `sudo … uvicorn …` start
command. See **Reboot behaviour (FileVault)** below for what "automatic" means
on this machine.

## One-time setup

### 1. DNS (mijn.host)
Set a single record and nothing else:

| Record | Type | Value |
|---|---|---|
| `lint-ii.valkuil.net` | A | `85.215.105.128` |

No `origin` record, no DDNS. (If `lint-ii.valkuil.net` still points at the home
IP, change it to `85.215.105.128` as the last step below, after testing.)

### 2. Tunnel key (Mac)
Create a dedicated passphraseless key so launchd can connect unattended:

```
ssh-keygen -t ed25519 -N "" -f ~/.ssh/lint-tunnel_ed25519 -C lint-ii-tunnel
```

Authorize it on Strato, **restricted to only this reverse tunnel**, by adding
one line to `antalb@85.215.105.128:~/.ssh/authorized_keys`:

```
restrict,port-forwarding,permitlisten="127.0.0.1:8000" ssh-ed25519 AAAA…(contents of lint-tunnel_ed25519.pub)… lint-ii-tunnel
```

`restrict` drops pty/agent/X11; `port-forwarding` + `permitlisten` allow *only*
the `127.0.0.1:8000` reverse forward and nothing else.

### 3. Strato sshd (recommended)
So a dead tunnel is reaped and port 8000 is freed promptly for the reconnect,
add to `/etc/ssh/sshd_config` and `systemctl reload ssh`:

```
ClientAliveInterval 30
ClientAliveCountMax 3
```

### 4. Mac launchd jobs
```
cp deploy/mac/net.valkuil.lint-ii.plist        ~/Library/LaunchAgents/
cp deploy/mac/net.valkuil.lint-ii-tunnel.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/net.valkuil.lint-ii.plist
launchctl load -w ~/Library/LaunchAgents/net.valkuil.lint-ii-tunnel.plist
```
Check: `curl -s http://127.0.0.1:8000/health` on the Mac → `{"status":"ok",…}`.

### 5. Strato nginx
```
sudo cp deploy/strato-nginx-lint-ii.conf /etc/nginx/sites-available/lint-ii
sudo ln -sf /etc/nginx/sites-available/lint-ii /etc/nginx/sites-enabled/lint-ii
sudo nginx -t && sudo systemctl reload nginx
```
With the tunnel up, on Strato: `curl -s http://127.0.0.1:8000/health` should
also return ok (proves the tunnel reaches home).

### 6. Test end-to-end BEFORE flipping public DNS
Pretend DNS already points at Strato (works from anywhere, changes no public
DNS):
```
curl -k --resolve lint-ii.valkuil.net:443:85.215.105.128 https://lint-ii.valkuil.net/health
curl -k --resolve lint-ii.valkuil.net:443:85.215.105.128 https://lint-ii.valkuil.net/editor_demo.html
```
Both should succeed via Strato → tunnel → home.

### 7. Flip public DNS
Lower the `lint-ii.valkuil.net` TTL ahead of time if you can, then point
`lint-ii.valkuil.net A → 85.215.105.128`. Verify from a network that previously
failed (the hotspot): open `https://lint-ii.valkuil.net/editor_demo.html` and
run an analysis.

### 8. Retire the old direct-to-home server
The pre-tunnel setup ran uvicorn manually with TLS on `0.0.0.0:8443` (started via
`sudo`). Once the new path is confirmed, stop it so there is a single server and
no chance of two model inferences competing for the 32 GB of RAM. On the Mac
(it is root-owned, so `sudo`); the match string hits only the old server because
the new one binds `127.0.0.1`, not `0.0.0.0`:
```
sudo pkill -f "uvicorn api:app --host 0.0.0.0"
```

## Reboot behaviour (FileVault)

This Mac has **FileVault on**, which is mutually exclusive with macOS automatic
login — so the demo cannot recover from a cold boot with zero interaction. What
*does* happen:

- After a reboot the Mac stops at the FileVault unlock screen. Entering the
  `antalb` password there both unlocks the disk **and logs that user into the
  desktop session**, which is what starts the two LaunchAgents (uvicorn +
  tunnel). So: **the demo comes back as soon as someone enters the password
  once after a reboot** — it does not stay down waiting for a separate login.
- It will *not* come back while the machine sits at the unlock screen
  untouched (e.g. after an unattended power loss). That is the price of
  FileVault and is accepted here.
- For a **planned remote reboot**, use `sudo fdesetup authrestart`: it takes the
  unlock credentials now and reboots straight past the FileVault screen into the
  logged-in session, so the agents start without anyone at the keyboard. (Only
  works for that one planned restart, not unexpected ones.)

## Operations

- **Is it up?** `launchctl list | grep lint-ii` on the Mac (both labels should
  show a PID). Logs in `~/Library/Logs/lint-ii*.log`.
- **502 from the site** = tunnel down or server down. The launchd jobs
  self-heal within ~15–90s; if it persists, check the tunnel log for auth/port
  errors and confirm `ClientAliveInterval` freed port 8000 on Strato.
- **Restart the server** after a code update:
  `launchctl kickstart -k gui/$(id -u)/net.valkuil.lint-ii`
- **Rollback** to direct-to-home (emergency): point `lint-ii.valkuil.net A` back
  at the home IP and run uvicorn with TLS on the externally forwarded port as
  before. Only needed if Strato itself is unavailable.

## TLS / certificate notes

- **Strato** holds the only public cert (`lint-ii.valkuil.net`). With DNS
  pointing here, certbot renews it on this box via the HTTP-01 ACME path in the
  config (`/.well-known/acme-challenge/`, webroot `/var/www/certbot` — adjust to
  your certbot setup; `--nginx` works too). Keep port 80 open.
- **Home** needs no cert at all (plain HTTP on loopback, behind SSH).
