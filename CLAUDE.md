# LiNT-II — notes for Claude Code sessions

Dutch readability analysis (LiNT score, spaCy `nl_core_news_lg`) + LLM-generated
readability suggestions with an interactive editor. Backend `api.py` (FastAPI),
suggestion engine `src/lint_ii/llm/`, frontend `src/visualizer/` +
`editor_demo.html`. Live demo: https://lint-ii.valkuil.net (Strato box, systemd).

## Two working sides — coordination

Sessions run on Antal's Mac (`~/Software/lint_ii`) and on the Strato box
(`antalb@85.215.105.128`, `~/servers/lint_ii`). **Both push to `fork` main
(antalvdb/lint_ii): always `git fetch fork && git rebase fork/main` before
pushing.** Don't assume the other side is idle.

- **Box side** has: the live service, `journalctl -u net.valkuil.lint-ii`,
  provider keys in `/etc/lint-ii/lint-ii.env` (`MISTRAL_API_KEY`,
  `HETZNER_API_KEY` — never commit), instant deploys. Prompt-iteration and
  deploy/verify work belongs here.
- **Mac side** has: the historical eval result files (gitignored), Claude's
  project memory, the py311 conda env for local pipeline checks
  (`/Users/antalb/opt/miniconda3/envs/py311/bin/python`, `sys.path` → `src`).

## Box operations

- Deploy: `git pull` + `sudo systemctl restart net.valkuil.lint-ii`
  (kickstart-style restarts do not re-read the env file; systemd restart does).
- Current live provider: `LINT_PROVIDER=hetzner` → Qwen (Qwen3.6-35B-A3B-FP8),
  temperature 0.3, thinking OFF (`chat_template_kwargs enable_thinking:false` —
  thinking ON burns max_tokens and yields 0 suggestions; see providers.py).
- Result cache: disk-persisted (`~/.cache/lint-ii/result_cache.json`), survives
  restarts BY DESIGN; the key includes the running git commit + model name, so
  deploys invalidate naturally and startup re-warms the example texts.
- Hetzner rate limits are output-bound (60k tokens/60s): sequential eval runs
  are safe, parallel-3 can brush the cap. Some non-prose inputs 422 (expected).
- Token accounting: every provider call logs one `LLM_USAGE` line at INFO
  (counts + model only, no text, so it is safe to leave on). Hetzner has no
  usage endpoint (`/api/v1/usage` is 404), so this log is the only record.
  Monthly totals:
  `grep -oP 'LLM_USAGE .*\btotal=\K\d+' /var/log/lint-ii/app.log | paste -sd+ | bc`
  Before 2026-09-11 nothing was logged; usage for Jul–Sep 2026 could only be
  ESTIMATED (~6,200 API calls ≈ 2–4M tokens/month, mostly eval runs and probe
  traffic rather than testers). A provider returning no usage block logs
  `total=?`, so gaps are visible instead of silently summing as zero.
- TLS (nginx + Let's Encrypt, `lint-ii.valkuil.net`): renewal is automatic via
  `certbot.timer`, with a deploy hook at
  `/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh` that runs
  `nginx -t && systemctl reload nginx`. **Keep that hook**: without it certbot
  writes the renewed cert but nginx serves the old one from memory until
  something restarts it, so the site expires anyway. The cert died once
  (2026-08-23) because it had been issued with `authenticator = manual` +
  `pref_challs = dns-01`, which certbot cannot renew unattended — the timer ran
  daily and silently skipped it for three months. It is now `authenticator =
  nginx`; if you ever re-issue, do NOT go back to a manual challenge. The
  single-name cert has no wildcard, so http-01 is fine.
  Diagnose with: `echo | openssl s_client -connect lint-ii.valkuil.net:443
  -servername lint-ii.valkuil.net 2>/dev/null | openssl x509 -noout -dates`,
  then `sudo certbot renew --dry-run`. Real errors are in
  `/var/log/letsencrypt/` (root-only).

## Hard-won rules — do not undo

- `/analyze` is job+poll (`POST` returns `job_id`, client polls
  `/analyze-result/{id}`). Never collapse it back into one long request:
  iOS WebKit aborts long requests and mobile breaks.
- Keep the `Cache-Control: no-cache` middleware for HTML responses, and bump
  the `?v=N` query strings on EVERY frontend JS/CSS change (they're the only
  cache busting).
- Hunspell `suggest()` is skipped for words > 14 chars (`_SUGGEST_MAX_LEN`) —
  spylls explodes combinatorially on long unknown compounds (minutes of CPU,
  invisible to all watchdogs). Don't raise it casually.
- Hard constraints on LLM output need DETERMINISTIC post-filters, not prompt
  lines (conjunction-split guard, URL guard, invented-content guard, relation
  whitelist, `_correction_plausible`, the word-family guard). Prompt-only rules
  were tried and failed. Related lesson: rhetorical questions in a
  structured-output prompt make Qwen answer in prose and break block parsing.
- When a prompt rule IS the right tool, Qwen moves on WORKED EXAMPLES, not on
  abstract guidance. `8397cae` fixed the connective gevolg gap with a delay and
  a measure example; the same session proved the converse by testing the
  rewritten guard clause on its own (the "simpele opeenvolging" narrowing that
  `0041a47` was built around) — scored identically to base. `0041a47`'s
  rhetorical question was correctly blamed for the collapse, but its underlying
  premise, that the clause suppressed temporal consequences, was simply wrong.
  Don't re-derive it: reach for an example before reaching for a sentence of
  explanation.
- Prompt examples LEAK LEXICALLY into nearby inputs. A connective example
  opening "De zaal was tot de laatste stoel gevuld" knocked corpus5 conn-4
  ("De zaal was ... toch uitverkocht") from 5/5 to 1/5; rewording restored 6/6.
  Keep new example vocabulary clear of corpus/tester text — a shared common
  noun is usually fine, a repeated phrase is not — and check with
  `python3 scripts/eval/connective_probe.py --audit`.
- "Not in SUBTLEX/Hunspell" does NOT mean "not a word" — Dutch productive
  compounds are routinely missing from both while being perfectly correct.
- Before trusting any eval delta or live probe: verify the fix is actually
  deployed (git HEAD on the box + fresh service start + a probe). Stale-deploy
  false conclusions have cost multiple debug cycles.

## Feature gates & env (set in /etc/lint-ii/lint-ii.env on the box)

- `LINT_II_CONNECTIVES=1` — the connective pass is OFF by default; the live box
  has it on. Any local engine experiment that should include connectives needs
  this set, or the pass silently returns nothing.
- `LINT_II_SWAP_JUDGE=1` — verification pass that drops meaning-changing
  word_frequency swaps (backlog item 2). OFF by default in code, and the box
  env file sets it to 0, so it is inactive in production (turned off after the
  set-4 run: ~2 false alarms per ~40 swaps whatever the wording, and each
  prompt fix relocated the errors rather than removing them). It only ever removes
  suggestions: measured on set 5 it dropped 3 meaning-changing swaps and 1
  legitimate one, and a false alarm on a single-suggestion item costs the whole
  item. Turn it on deliberately, and re-read the calibration table in
  `scripts/eval/README.md` before changing its prompt.
- Other knobs: `LINT_II_LLM_TIMEOUT` (watchdog, 300s), `LINT_II_MAX_PENDING_JOBS`
  (flood guard, 12), `LINT_II_HETZNER_TEMPERATURE` (0.3),
  `LINT_II_LOG_LEVEL` (INFO; DEBUG logs full prompts+responses — the fastest way
  to see raw Qwen output during prompt iteration, but don't leave it on: tester
  texts would pile up in the logs).
- `LINT_CONSOLIDATE_REWRITES` — consolidated per-sentence rewrites, default ON.

## Product context (why this exists)

Concept test of LiNT-driven LLM readability suggestions with academic testers
(Henk Pander Maat — LiNT's author, Merel Scholman, gemeente staff). Their open
feature asks, in Antal's order of interest: editable/intermediate splits (a
2-sentence middle option; suggestions editable in place), a tone/genre
selector, interaction logging (needs a consent checkbox). The tool's privacy
positioning matters: tester texts can contain PII, and the cloud providers see
whatever is analyzed — flag this before analyzing real letters.

## Eval harness

Self-diagnosis unit in `scripts/eval/` — see `scripts/eval/README.md` for the
workflow, corpus inventory, cross-set results and the current backlog.
From the box, run it against `--base http://127.0.0.1:8000` (bypasses the
nginx edge rate limits). Judging is LLM-as-judge: Claude reads the results
file and scores each suggestion wrong / debatable / right.

## Conventions

- Commit messages: conventional-commit style (`fix(llm): ...`), body explains
  the why; end with the Claude co-author trailer when Claude authors.
- On the Mac, `git commit --no-gpg-sign` (Antal's signing key expired).
