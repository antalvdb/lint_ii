# Deployment: Strato 443 reverse proxy → home Mac Studio

## Problem this solves

The demo runs on the Mac Studio at home on **port 8443** (the home ISP blocks
inbound 443, so the router forwards 8443). Restrictive client networks — mobile
hotspots, corporate/guest WiFi, some carriers — only allow outbound **80/443**,
so testers on those networks see *"the network did not allow a connection"*.

Fix: the Strato Linux box (`85.215.105.128`, a data-center host) terminates TLS
on the standard **443** — reachable on virtually every network — and reverse-
proxies to the home Mac Studio, which keeps doing the (fast) inference.

```
client ──443──▶ Strato (85.215.105.128) ──8443──▶ home Mac Studio (inference)
```

No application code changes are involved: the frontend uses only relative URLs
(`/analyze-lint`, `/analyze`, `/analyze-result/...`) and the API hardcodes no
host/port, so it works unchanged behind the proxy.

## DNS (mijn.host)

| Record | Type | Value | Notes |
|---|---|---|---|
| `lint-ii.valkuil.net` | A | `85.215.105.128` | the Strato box (was: home IP) |
| `origin.valkuil.net` | A | *home router public IP* | **new**; point the existing DDNS updater at THIS record now, not `lint-ii` |

The home IP is dynamic. Whatever mechanism kept `lint-ii.valkuil.net` updated to
the home IP must now update `origin.valkuil.net` instead. If the home IP changes
and `origin` is not updated, the proxy breaks. (nginx re-resolves `origin` on a
30s TTL via the `resolver` line, so once the record is corrected it recovers
without an nginx reload.)

## Cutover (do in this order — avoids downtime)

1. **Add `origin.valkuil.net` A → home public IP** in mijn.host, and repoint the
   DDNS updater at it. Wait for it to resolve:
   `dig +short origin.valkuil.net` should return the home IP.

2. **Confirm home is reachable through it** (from the Strato box):
   `curl -k https://origin.valkuil.net:8443/health` → `{"status":"ok",...}`.

3. **Install the nginx config on Strato**: copy `strato-nginx-lint-ii.conf` to
   `/etc/nginx/sites-available/lint-ii`, ensure the `sites-enabled` symlink,
   then `sudo nginx -t && sudo systemctl reload nginx`.

4. **Test end-to-end BEFORE flipping public DNS**, by pretending DNS already
   points at Strato (works from anywhere, doesn't change public DNS):
   `curl -k --resolve lint-ii.valkuil.net:443:85.215.105.128 https://lint-ii.valkuil.net/health`
   then the same for `/editor_demo.html`. Both should succeed via Strato→home.

5. **Lower the `lint-ii.valkuil.net` TTL** (e.g. 300s) a while ahead if you can,
   then **flip `lint-ii.valkuil.net` A → 85.215.105.128**.

6. **Verify from a network that previously failed** (the hotspot): open
   `https://lint-ii.valkuil.net/editor_demo.html` and run an analysis.

## Rollback

Point `lint-ii.valkuil.net` A back to the home public IP. This instantly
restores the previous direct-to-home:8443 setup. (Keep the DDNS updater able to
maintain whichever record is currently the public one.)

## TLS / certificate notes

- **Strato** already has a Let's Encrypt cert for `lint-ii.valkuil.net`. With DNS
  now pointing here, certbot renews it on this box via the HTTP-01 ACME path in
  the config (`/.well-known/acme-challenge/`, webroot `/var/www/certbot` — adjust
  to your certbot setup; if you use `--nginx`, that works too). Keep port 80 open.
- **Home** cert renewal stops (DNS no longer resolves there for the challenge).
  That's fine: nginx proxies upstream with `proxy_ssl_verify off`, so an expired
  or mismatched home cert still works. Optionally switch the home server to a
  self-signed cert later; not urgent.

## Unchanged

- Home Mac Studio: same uvicorn + TLS on 8443, same router port-forward
  (8443 → 192.168.178.95:8443). No change.
- The Strato box's old `uvicorn` on 127.0.0.1:8080 is no longer used by this
  config (we proxy to home instead); leave it stopped.
