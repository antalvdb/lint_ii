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
- "Not in SUBTLEX/Hunspell" does NOT mean "not a word" — Dutch productive
  compounds are routinely missing from both while being perfectly correct.
- Before trusting any eval delta or live probe: verify the fix is actually
  deployed (git HEAD on the box + fresh service start + a probe). Stale-deploy
  false conclusions have cost multiple debug cycles.

## Eval harness

Self-diagnosis unit in `scripts/eval/` — see `scripts/eval/README.md` for the
workflow, corpus inventory, cross-set results and the current backlog.

## Conventions

- Commit messages: conventional-commit style (`fix(llm): ...`), body explains
  the why; end with the Claude co-author trailer when Claude authors.
- On the Mac, `git commit --no-gpg-sign` (Antal's signing key expired).
