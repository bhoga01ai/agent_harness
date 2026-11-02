# ADK search agent + Harness²

A Google ADK agent that answers questions with Google Search. Its harness is improved by
[Harness²](https://github.com/google-research/harness2), which is used unmodified as a library.
This project adds Google ADK as a third Harness² executor (a "substrate"), next to OpenCode and
Codex.

## How it fits together

```
improve_harness.py ──> harness loop: run_domain()  (Harness² core)
                         for each task: propose -> probe -> compose -> select -> remember
                         ├─ adapter.py   holds the tasks, runs your agent, grades answers
                         └─ proposer.py  AI editor (Claude Code) that edits harness copies
                       writes runs/loop/<model>/{executions,runs,audit}
                       ends with a scoreboard and the best harness
```

| Piece | File | Role |
|---|---|---|
| Command | `harness_loop/improve_harness.py` | You run it; it runs the loop, the scoreboard and picks the best harness |
| Adapter | `harness_loop/adapter.py` | The questions in `tasks/search_tasks.json`; runs the agent on a harness copy; grades answers |
| Proposer | `harness_loop/proposer.py` | The editor: Claude Code sessions on your local login, and what each harness file means in ADK |
| Your agent | `search_agent/agent.py` | What `adk web` serves (`harness/active/`, or `harness/base/` if there is none); also the harness's 8 parts (`ADK_SURFACE`) and `build_agent`, which the loop uses on each edited copy |
| Your harness | `harness/base/` | The editable parts the loop proposes changes to |

How each harness file becomes part of the ADK agent:

| Harness file | In ADK |
|---|---|
| `systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`, `tool_descriptions/tool_guidance.md` | The root agent's instruction |
| `sub_agents/<name>/AGENT.md` | An `Agent` with `google_search`, called as an `AgentTool` |
| `skills/<name>/SKILL.md` | Loaded on demand by the `load_skill` tool |
| `scripts/*.py` | Run by the `run_script` tool |
| `hooks.json` | `before_tool_callback` / `after_tool_callback` |

## Running the harness loop

### Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey), or a Google
  Cloud project with Vertex AI enabled. The agent and the grader use it.
- The [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI, installed and logged in
  (`claude` on your `PATH`). The editor and selector run as Claude Code sessions.
- `python3` on your `PATH`, for the harness hook in `hooks.json`.

### Step 1: Install

```bash
uv sync
```

### Step 2: Add your credentials

```bash
cp .env.example .env
```

Open `.env` and fill in one option:

- **Gemini API key:** set `GOOGLE_API_KEY`, and keep `GOOGLE_GENAI_USE_ENTERPRISE=FALSE`.
- **Vertex AI:** run `gcloud auth application-default login` first, then set
  `GOOGLE_GENAI_USE_ENTERPRISE=TRUE`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

Optional settings in the same file:

| Variable | Default | What it sets |
|---|---|---|
| `HARNESS2_ADK_MODEL` | `gemini-2.5-flash` | The agent's model. It also names the output folder, `runs/loop/<model>/`. |
| `HARNESS2_IMPROVER_MODEL` | `claude-sonnet-5-5` | The editor and selector model |
| `HARNESS2_JUDGE_MODEL` | `gemini-2.5-flash` | The model that grades answers against the references |
| `HARNESS2_CLAUDE_VERTEX` | unset | Set to `1` to run Claude Code on Vertex AI instead of your local login |

### Step 3: Validate the harness and preview the run

```bash
uv run improve-harness --dry-run --max-tasks 1
```

This makes no model calls. It first validates `harness/base/` with the same rules the loop
applies to every edited harness, and builds the agent from it:

- the always-on files are not empty and stay under 24,000 characters;
- every skill has `name:` matching its folder and a `description:`, and is named in
  `systemprompt.md` or `guardrails.md`;
- every sub-agent has a `description:`, and `sub_agents/searcher/` exists;
- every script compiles, and every script named in a skill or always-on file exists;
- `hooks.json` is valid JSON with only `PreToolUse` / `PostToolUse` and `"command"` hooks.

Then it prints the tasks it would run, grouped by topic area, the number of agent rollouts and
editor sessions, and the output folder:

```
Harness ok: 9 files, 1471 always-on chars, 1 sub-agent(s), 1 skill(s), 2 script(s), a hooks file; agent tools: searcher, load_skill, run_script
3 task(s), k=1, parallel mode: about 12 agent rollouts and 12 editor sessions -> runs/loop/gemini-2.5-flash
  history: history-01
  science: science-01
  tech: tech-01
```

If the harness is invalid, it lists every problem and exits with code `1`, for example:

```
Harness harness/base is invalid:
  - skill 'reconcile-sources' is never named in the always-on text, ...
  - scripts/bad.py does not compile: SyntaxError: invalid syntax (line 1)
```

A real run does the same check first, so it never spends model calls on an invalid harness.
Validation checks the harness's shape, not whether it makes the agent answer better; that
takes a real run.

### Step 4: Run the loop on one task

Start with one task that has a reference answer, so the scoreboard can grade it:

```bash
uv run improve-harness --only-tasks science-01
```

The command first checks that your credentials and the `claude` CLI are set up. For each task,
it then:

1. runs the agent on the base harness (`harness/base/`);
2. has the editor propose changes to copies of the harness, using the agent's trajectories;
3. runs the agent on each edited copy (probe), and combines the best edits (compose);
4. has the selector pick one harness, judging the trajectories, not the reference answers;
5. saves lessons for later tasks in the same topic area (remember).

The loop prints one line per finished task, then a scoreboard:

```
Scoreboard: base 0/1 -> adopted 1/1
  science-01: base 0 -> adopted 1

Best harness (from science-01): runs/loop/gemini-2.5-flash/best_harness
  why: ...
  It scored higher than base. Re-run with --adopt to serve it in `adk web`.
```

### Step 5: Review what changed

Everything is under `runs/loop/<model>/`:

| Path | What |
|---|---|
| `best_harness/` | The edited harness that gained most over base. Read it before adopting it. |
| `scoreboard.json` | Base and adopted grades per task, the selected harness and the selector's reason |
| `executions/` | Every agent answer: `answer.md`, `events.jsonl` (the trajectory), `meta.json` |
| `runs/` | Every edited harness and the selection decisions (`summary.json` per task) |
| `runs/_library/` | Lessons carried forward to later tasks in the same topic area |
| `audit/` | Every editor session |

To see exactly what changed, compare the best harness with the base:

```bash
diff -ru harness/base runs/loop/gemini-2.5-flash/best_harness
```

### Step 6: Adopt the best harness

```bash
uv run improve-harness --only-tasks science-01 --adopt
```

Use the same options as the run you are adopting. Finished work is reused, so this costs only
the grading calls. The best harness is copied into `harness/active/` only if it scored higher
than base. Any previous `harness/active/` is moved to `harness/history/<timestamp>/`.

### Step 7: Chat with the improved agent

```bash
uv run adk web .
```

Open the URL it prints and pick `search_agent`. It serves `harness/active/` when there is one,
and `harness/base/` otherwise. Restart `adk web` after adopting a new harness. To go back to
the base harness, delete `harness/active/`.

### Step 8: Run all the tasks

```bash
uv run improve-harness --k 1 --mode parallel
```

The three topic areas run at the same time (`--workers 3`). Within a topic area, tasks run in
order, so later tasks benefit from what earlier ones learned; each topic area keeps its own
lessons, so running them side by side changes nothing about the results. With 6 tasks, a run
takes about 7 minutes instead of about 20. If a run stops partway, run the same command again:
finished rollouts, editor sessions and selections are reused.

### Options

| Option | Default | What it does |
|---|---|---|
| `--k N` | `1` | Scale of the search. Each task costs about `4N` agent rollouts and `4N` editor sessions. |
| `--mode` | `parallel` | `parallel` or `sequential`: how the editor's proposals are combined |
| `--only-tasks a,b` | all | Run only these task ids from `tasks/search_tasks.json` |
| `--max-tasks N` | all | Run at most N tasks per topic area |
| `--workers N` | `3` | Topic areas to run at the same time. Use `1` to run them one after another. |
| `--dry-run` | off | Validate the harness, print the tasks and the cost, then stop |
| `--adopt` | off | Copy the best harness into `harness/active/` if it scored higher than base |

The output folder depends on `--k`, `--mode` and the agent model. Changing any of them starts
a fresh run instead of reusing the previous one.

### Tasks and reference answers

The questions are in `tasks/search_tasks.json`. Each one has an `id`, a `domain` (the topic
area) and a `question`, and optionally a `reference` answer:

```json
{"id": "science-03", "domain": "science", "question": "...", "reference": "..."}
```

The agent, the editor and the selector never see the references. They are read only at the
end, when Gemini grades the base and adopted answers for the scoreboard. A task without a
reference shows `no reference` and cannot count towards adoption. If the grading call fails,
the task shows `not graded: the judge call failed` and is left out of the totals, not counted
as wrong.

### Troubleshooting

| Message | Fix |
|---|---|
| `Harness ... is invalid` | Fix each listed problem in `harness/base/`, then run `--dry-run` again. |
| `GOOGLE_API_KEY is unset` | Do step 2: copy `.env.example` to `.env` and add your key. |
| `Vertex AI is selected but GOOGLE_CLOUD_PROJECT is unset` | Set `GOOGLE_CLOUD_PROJECT` in `.env`. |
| `the claude CLI is not on PATH` | Install Claude Code and run `claude` once to log in. |
| `only_tasks names unknown task id(s)` | Use an `id` from `tasks/search_tasks.json`. |
| `not graded: the judge call failed` | The Gemini grading call errored. Check `GOOGLE_API_KEY` and `HARNESS2_JUDGE_MODEL`, then run the same command again: only the grading is redone. |
| `No edited harness was selected` | The selector kept the base harness for every task. Try more tasks or a higher `--k`. |
| `It did not score higher than base` | Nothing was adopted. Review `best_harness/` and `scoreboard.json` to see why. |

The command exits with code `2` when at least one task failed. The per-task lines show the
error, and `runs/` holds the details.

## Safety

- Harness² runs each editor session as `claude --dangerously-skip-permissions` in a temporary
  folder.
- Scripts and hooks that the editor writes run on this machine during rollouts, from a
  temporary copy of the harness. Read a harness before you adopt it.
