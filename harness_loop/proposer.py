"""The AI editor: Claude Code sessions that edit copies of the harness.

Each session gets a copy of the harness and the agent's trajectories and edits
the copy; Harness² probes and selects among the edits. SURFACE_ADK tells the
editor what each harness file becomes in ADK. Harness²'s `claude_cli` channel
always switches Claude Code to Vertex AI; `LocalClaudeProposer` uses the Claude
Code login already on this machine instead, unless HARNESS2_CLAUDE_VERTEX=1.
"""

from __future__ import annotations

import os

from harness2.core import prompts
from harness2.core.proposer import Proposer

# Set by a parent Claude Code session; a child session must not inherit them.
_PARENT_SESSION_ENV = ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CODE_SSE_PORT')


class LocalClaudeProposer(Proposer):

  def env(self) -> dict[str, str]:
    if os.environ.get('HARNESS2_CLAUDE_VERTEX') == '1':
      return super().env()
    e = {k: v for k, v in os.environ.items() if k not in _PARENT_SESSION_ENV}
    e.pop('CLAUDE_CODE_USE_VERTEX', None)
    e['ANTHROPIC_MODEL'] = self.spec.model
    return e


SURFACE_ADK = """\
# WHAT YOU CAN CHANGE

The harness is a tree of components that is assembled into a Google ADK (Agent Development Kit)
agent before every run. Each component has a different runtime cost and a different reach:

| path | what it becomes in ADK | when it is in context |
|---|---|---|
| `systemprompt.md` | part of the root agent's instruction | ALWAYS ON -- every model call |
| `guardrails.md` | part of the root agent's instruction | ALWAYS ON -- same cost |
| `LongTermMEMORY.md` | part of the root agent's instruction | ALWAYS ON -- same cost |
| `tool_descriptions/tool_guidance.md` | part of the root agent's instruction | ALWAYS ON -- same cost |
| `skills/<name>/SKILL.md` | a body returned by the `load_skill` tool | name+description listed; body loaded ON DEMAND |
| `skills/<name>/scripts/*` | a helper carried with its skill | run ON DEMAND with the `run_script` tool |
| `sub_agents/<name>/AGENT.md` | an ADK sub-agent with `google_search`, exposed as tool `<name>` | runs in its OWN context, returns text |
| `scripts/<name>.py` | a deterministic helper | run ON DEMAND with the `run_script` tool |
| `hooks.json` | ADK before/after tool callbacks | fire INSIDE the tool loop, on every matching tool call |

The four that are easy to get wrong:

`sub_agents/` -- frontmatter needs a `description:`; the body is the sub-agent's whole
instruction. The root agent calls it as a tool named after the directory (hyphens become
underscores), with ONE argument, `request` (a string). The sub-agent sees only its AGENT.md body and that request, never the
conversation, and it can do one thing: run Google Search. The base harness ships
`sub_agents/searcher/` -- it is the ONLY way this agent reaches the web, so deleting it leaves an
agent that cannot search. Add another sub-agent only for a search job whose instruction differs
(for example one that only reads official sources).

`skills/` -- frontmatter needs `name:` matching the directory and a `description:`, or it is
dropped silently. The agent sees each skill's name and description and calls
`load_skill(name)` to read the body.

`scripts/` -- run as `run_script(path="scripts/<name>.py", args=[...])` from a working copy of
the harness, Python standard library only, with no network access assumed and a 60 second limit.
The tool returns the exit code, stdout and stderr.

`hooks.json` -- Claude-Code hook schema: a top-level "hooks" object maps `PreToolUse` or
`PostToolUse` to entries of {"matcher": "<regex on the tool name>", "hooks": [{"type": "command",
"command": "..."}]}. Tool names are `load_skill`, `run_script` and one per sub-agent (e.g.
`searcher`). Each command runs from the harness working copy and receives the event as JSON on
stdin (`tool_name`, `tool_input`; PostToolUse adds `tool_response`). Printing
{"decision": "block", "reason": "..."} from a PreToolUse hook cancels the call and returns the
reason to the model as the tool result; from a PostToolUse hook it attaches the reason to the
tool result as `hook_feedback`. The only component that acts in the tool loop rather than in the
prompt. Keep commands fast; put real logic in a `scripts/` helper the hook calls.

Change only what a trajectory shows needs changing -- but WHERE you change something, the FORM
matters as much as the content, and the cheap form is not the obvious one.

ALWAYS-ON PROSE IS THE EXPENSIVE OPTION. Every word of it is in the instruction on every model
call. An on-demand component costs nothing until the moment it is invoked.

So prefer, in this order:
  - a SKILL for anything with a recognisable trigger -- how to research a kind of question, how
    to reconcile sources that disagree, what a complete answer of a given kind contains.
  - a SUB-AGENT instruction change when the searcher's results were the problem -- it returned
    too little, the wrong kind of source, or facts without URLs.
  - a SCRIPT for a deterministic operation the runs hand-rolled and got wrong (date
    arithmetic, unit conversion).
  - a HOOK when a mechanical problem in the tool loop recurs -- blocking a known-bad call with a
    reason that teaches the fix, or checking every searcher result for URLs.
  - ALWAYS-ON TEXT last, and only for a short rule that genuinely applies to every question.
    If you add always-on text, cut something else to pay for it.

THREE RULES THAT DISCARD THE WHOLE PROPOSAL. Validation runs before any rollout, and a harness
that fails it is thrown away entire.

1. EVERY SKILL NEEDS A TRIGGER LINE. If you create or keep `skills/<name>/SKILL.md`, its
   directory name must appear VERBATIM in `systemprompt.md` or `guardrails.md`, in the SAME
   edit:

       When <the situation this skill handles>, call `load_skill` with `<name>`.

2. EVERY SCRIPT YOU NAME MUST EXIST. A `scripts/<file>.py` named from a SKILL.md or from
   always-on text must ship in the same proposal; a skill's own helper goes at
   `skills/<skill-name>/scripts/<file>.py`.

3. `hooks.json` MUST BE VALID JSON in the schema above, with only PreToolUse and PostToolUse
   events and "command"-type hooks.

THE ANSWER'S SHAPE BELONGS TO THE REQUEST, NOT TO YOU. A harness changes HOW the agent works; it
must never change WHAT the person asked for. If the question asks for a number, a list or a
yes/no, that is the answer's shape.

""" + prompts._surface_part(prompts._SURFACE_OPENCODE, '# HOW TO MAKE A CHANGE')


def register() -> None:
  """Adds the "adk" surface text to Harness²'s prompt table (idempotent)."""
  if 'adk' in prompts._SURFACE_TEXTS:
    return
  prompts._SURFACE_TEXTS['adk'] = prompts._SurfaceText(
      surface_md=SURFACE_ADK,
      inv=prompts._INV_TEMPLATE.replace(
          '__LOOP_LEVER__',
          '`hooks.json` PreToolUse or PostToolUse hook -- for example, rejecting a searcher'
          ' request that bundles several questions, with a reason that says to split it',
      ),
      dom=prompts._DOM_TEMPLATE.replace(
          '__TRIGGER_SENTENCE__',
          'A skill exists only once its name appears in the always-on text: write that'
          ' trigger line in the same edit.',
      ),
      tts_surface=SURFACE_ADK,
  )


register()
