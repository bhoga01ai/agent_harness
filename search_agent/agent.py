"""The ADK search agent served by `adk web` / `adk run`, built from a harness.

`root_agent` is built from harness/active (a harness you adopted with
`improve-harness --adopt`), falling back to harness/base. The harness loop
builds the same agent from each edited copy with `build_agent`.

The harness's 8 parts, and what each becomes in ADK:

    always-on files      -> the root agent's instruction
    sub_agents/<n>/      -> an Agent with google_search, wrapped as AgentTool `<n>`
    skills/<n>/SKILL.md  -> listed by name; body returned by the `load_skill` tool
    scripts/*.py         -> run by the `run_script` tool
    hooks.json           -> before/after tool callbacks
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import AgentTool, BaseTool, ToolContext, google_search
from harness2 import config
from harness2.core import tree as harness_tree
from harness2.core.tree import SurfaceSpec

DEFAULT_MODEL = 'gemini-2.5-flash'
# Runtime fence owned by the substrate, not the editable harness.
MAX_TOOL_CALLS_PER_TURN = 12
SCRIPT_TIMEOUT_S = 60
HOOK_TIMEOUT_S = 20
_TOOL_COUNT_KEY = 'harness:tool_calls'
_SCRIPT_PATH = re.compile(r'(?:skills/[a-z0-9][a-z0-9_\-]*/)?scripts/[A-Za-z0-9_.\-]+')

# The editable tree Harness² proposes changes to; build_agent below assembles it.
ADK_SURFACE = SurfaceSpec(
    name='adk',
    always_on=(
        'systemprompt.md',
        'guardrails.md',
        'LongTermMEMORY.md',
        'tool_descriptions/tool_guidance.md',
    ),
    always_on_max=24_000,
    always_on_cap_reason=(
        "it is the agent's system instruction on every model call and competes with the"
        ' searcher results for context'
    ),
    skill_dir='skills',
    skill_name_rule='must_match',
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to systemprompt.md or guardrails.md,'
        ' e.g. "When <the situation this skill handles>, call `load_skill` with `{name}`."'
    ),
    autotrigger_file='guardrails.md',
    trigger_line='- When {desc}, call `load_skill` with `{name}`.\n',
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file='hooks.json',
    hook_events=frozenset({'PreToolUse', 'PostToolUse'}),
    compile_skill_scripts=True,
    reject_empty_always_on=True,
    stubs=(
        (
            'scripts/README.md',
            '<!-- c7: deterministic helpers, run with the `run_script` tool. Add one by'
            ' emitting scripts/<name>.py. -->\n',
        ),
        (
            'sub_agents/README.md',
            '<!-- c5: Google Search sub-agents, each called as a tool in its OWN context. Add'
            ' one as sub_agents/<name>/AGENT.md with a `description:` frontmatter line. -->\n',
        ),
        (
            'skills/README.md',
            '<!-- c6: workflows loaded on demand with the `load_skill` tool. Add one as'
            ' skills/<name>/SKILL.md with frontmatter `name:` matching the directory and a'
            ' `description:`. -->\n',
        ),
    ),
)


def solver_model() -> str:
  return os.environ.get('HARNESS2_ADK_MODEL') or DEFAULT_MODEL


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
  """Returns (frontmatter fields, body) for a `---` delimited markdown file."""
  m = re.match(r'^---\s*\n(.*?)\n---\s*\n?(.*)$', text, re.S)
  if not m:
    return {}, text
  fields = {}
  for line in m.group(1).splitlines():
    key, sep, value = line.partition(':')
    if sep:
      fields[key.strip()] = value.strip()
  return fields, m.group(2)


def tool_name(component: str) -> str:
  """A directory name as a Python identifier, which ADK requires of agent names."""
  name = component.replace('-', '_')
  return name if name.isidentifier() else f'agent_{name}'


def _instruction(tree: harness_tree.Tree):
  skills = []
  for name in harness_tree.skill_names(ADK_SURFACE, tree):
    meta, _ = split_frontmatter(tree[f'skills/{name}/SKILL.md'])
    skills.append(f"- `{name}`: {meta.get('description', '(no description)')}")
  scripts = sorted(rel for rel in tree if _SCRIPT_PATH.fullmatch(rel))

  parts = [
      body for f in ADK_SURFACE.always_on if (body := tree.get(f, '').strip())
  ]
  if skills:
    parts.append(
        '# AVAILABLE SKILLS\nCall `load_skill` with the name when the situation'
        ' matches:\n' + '\n'.join(skills)
    )
  if scripts:
    parts.append(
        '# HELPER SCRIPTS\nRun with `run_script`: ' + ', '.join(f'`{s}`' for s in scripts)
    )
  text = '\n\n---\n\n'.join(parts)

  # A function, so ADK does not treat {braces} in harness text as state placeholders.
  def instruction(ctx: ReadonlyContext) -> str:
    return f'{text}\n\nToday is {datetime.date.today().isoformat()}.'

  return instruction


def _make_tools(tree: harness_tree.Tree, workdir: Path, model) -> list:
  tools: list = []
  for name in harness_tree.subagent_names(ADK_SURFACE, tree):
    meta, body = split_frontmatter(tree[f'sub_agents/{name}/AGENT.md'])
    sub = Agent(
        name=tool_name(name),
        model=model,
        description=meta.get('description', f'The {name} sub-agent.'),
        instruction=body.strip() or meta.get('description', ''),
        tools=[google_search],
    )
    tools.append(AgentTool(agent=sub))

  if harness_tree.skill_names(ADK_SURFACE, tree):

    def load_skill(name: str) -> dict:
      """Returns the full instructions of a skill listed under AVAILABLE SKILLS.

      Args:
        name: The skill name exactly as listed.
      """
      body = tree.get(f'skills/{name}/SKILL.md')
      if body is None:
        return {'error': f'no skill named {name!r}'}
      return {'skill': name, 'instructions': split_frontmatter(body)[1]}

    tools.append(load_skill)

  if any(_SCRIPT_PATH.fullmatch(rel) for rel in tree):

    def run_script(path: str, args: list[str] | None = None, stdin: str = '') -> dict:
      """Runs a helper script from the harness and returns its output.

      Args:
        path: The script path as listed under HELPER SCRIPTS, e.g. scripts/x.py.
        args: Command-line arguments for the script.
        stdin: Text passed to the script on standard input.
      """
      rel = path.strip().lstrip('./')
      if not _SCRIPT_PATH.fullmatch(rel) or rel not in tree:
        return {'error': f'{path!r} is not a script in this harness'}
      try:
        p = subprocess.run(
            [sys.executable, '-I', rel, *(args or [])],
            cwd=workdir,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_S,
        )
      except subprocess.TimeoutExpired:
        return {'error': f'{rel} timed out after {SCRIPT_TIMEOUT_S}s'}
      return {
          'exit_code': p.returncode,
          'stdout': p.stdout[-20_000:],
          'stderr': p.stderr[-4_000:],
      }

    tools.append(run_script)
  return tools


def _tool_budget(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict | None:
  record = tool_context.state.get(_TOOL_COUNT_KEY) or {}
  count = record.get('count', 0) if record.get('invocation') == tool_context.invocation_id else 0
  if count >= MAX_TOOL_CALLS_PER_TURN:
    return {
        'error': (
            f'Tool budget of {MAX_TOOL_CALLS_PER_TURN} calls reached. Answer with what you'
            ' already have.'
        )
    }
  tool_context.state[_TOOL_COUNT_KEY] = {
      'invocation': tool_context.invocation_id,
      'count': count + 1,
  }
  return None


def _run_hooks(entries: list[dict], name: str, event: dict, workdir: Path) -> str:
  """Runs matching command hooks; returns the first block reason, or ""."""
  for entry in entries:
    matcher = entry.get('matcher') or '*'
    if matcher != '*' and not re.fullmatch(matcher, name):
      continue
    for hook in entry.get('hooks', []):
      try:
        p = subprocess.run(
            hook['command'],
            shell=True,
            cwd=workdir,
            input=json.dumps(event, default=str),
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_S,
        )
        out = json.loads(p.stdout or '{}')
      except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        continue
      if isinstance(out, dict) and out.get('decision') == 'block':
        return str(out.get('reason') or 'blocked by hook')
  return ''


def _hook_callbacks(tree: harness_tree.Tree, workdir: Path):
  try:
    hooks = json.loads(tree.get('hooks.json', '{}')).get('hooks', {})
  except json.JSONDecodeError:
    hooks = {}
  pre, post = hooks.get('PreToolUse', []), hooks.get('PostToolUse', [])

  def pre_tool(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict | None:
    event = {'hook_event_name': 'PreToolUse', 'tool_name': tool.name, 'tool_input': args}
    reason = _run_hooks(pre, tool.name, event, workdir)
    return {'error': reason} if reason else None

  def post_tool(
      tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: Any
  ) -> dict | None:
    event = {
        'hook_event_name': 'PostToolUse',
        'tool_name': tool.name,
        'tool_input': args,
        'tool_response': tool_response,
    }
    reason = _run_hooks(post, tool.name, event, workdir)
    if not reason:
      return None
    base = tool_response if isinstance(tool_response, dict) else {'result': tool_response}
    return {**base, 'hook_feedback': reason}

  return (pre_tool if pre else None), (post_tool if post else None)


def build_agent(
    harness_dir: Path | str,
    *,
    model: str | BaseLlm | None = None,
    workdir: Path | str | None = None,
) -> Agent:
  """The root ADK agent for one harness tree.

  `workdir` is where scripts and hooks run (a staged copy during rollouts);
  it defaults to the harness directory itself.
  """
  harness_dir = Path(harness_dir)
  tree = harness_tree.read_tree(ADK_SURFACE, harness_dir)
  if not tree:
    raise ValueError(f'{harness_dir} holds no harness files')
  workdir = Path(workdir or harness_dir)
  model = model or solver_model()
  pre_hook, post_hook = _hook_callbacks(tree, workdir)
  return Agent(
      name='search_agent',
      model=model,
      description='Answers questions using up-to-date web search.',
      instruction=_instruction(tree),
      tools=_make_tools(tree, workdir, model),
      before_tool_callback=[cb for cb in (_tool_budget, pre_hook) if cb],
      after_tool_callback=post_hook,
  )


PROJECT = Path(__file__).resolve().parent.parent
config.load_dotenv(PROJECT / '.env')

ACTIVE = PROJECT / 'harness' / 'active'
HARNESS_DIR = ACTIVE if harness_tree.is_tree(ADK_SURFACE, ACTIVE) else PROJECT / 'harness' / 'base'

root_agent = build_agent(HARNESS_DIR)
