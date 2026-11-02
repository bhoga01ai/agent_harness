# Tools

- `searcher(request)`: runs Google Search in its own context and returns facts as bullets with
  source titles and URLs. Put everything it needs in `request`; it cannot see this conversation.
- `load_skill(name)`: returns the full steps of a skill listed under AVAILABLE SKILLS.
- `run_script(path, args)`: runs a helper script. Use `scripts/date_diff.py` for any date
  arithmetic (how long ago, how old, days between two dates) instead of computing it yourself.
