"""The Harness² adapter: holds the tasks, runs the agent, grades the answers.

Each task is a question from tasks/search_tasks.json. A rollout runs the ADK
agent built from one harness copy and writes, under executions/<run id>/:

    events.jsonl   one record per text / tool call / tool result / error
    answer.md      the agent's final reply
    meta.json      run id, harness fingerprint, model, timing, status

The editor never sees the optional `reference` answers; only `score`, used
after selection, reads them.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types
from harness2.core import runmark
from harness2.core import tree as harness_tree
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec

from search_agent.agent import ADK_SURFACE, build_agent
from search_agent.agent import solver_model as default_solver_model

PROJECT = Path(__file__).resolve().parent.parent
TASKS_FILE = PROJECT / 'tasks' / 'search_tasks.json'
BASE_HARNESS = PROJECT / 'harness' / 'base'
DEFAULT_IMPROVER_MODEL = 'claude-sonnet-5-5'
ROLLOUT_TIMEOUT_S = 300
TRAJECTORY_MAX_CHARS = 200_000

PROMPTS = BenchPrompts(
    work=(
        'A person asks a question that needs current information from the web. The agent'
        ' delegates searches to Google Search sub-agents and replies with a direct answer,'
        ' the key supporting facts, and the sources it relied on.'
    ),
    domain_word='topic area',
    deliverable_word='answer',
    yardstick=(
        'A good run asks the searcher focused questions, checks that the facts it reports'
        ' are in the returned results, notices when sources disagree or are out of date,'
        ' and answers exactly what was asked with sources a reader can open.'
    ),
    domain_guidance=(
        'Questions are grouped by topic area (tech, science, history). What counts as a'
        ' trustworthy and current source differs between them: release notes and vendor'
        ' docs for tech, agencies and papers for science, contemporary reporting for'
        ' history.'
    ),
)


def improver_model() -> str:
  return os.environ.get('HARNESS2_IMPROVER_MODEL') or DEFAULT_IMPROVER_MODEL


class AdkSubstrate:
  """All Harness² core asks of a substrate: a name and the editable surface."""

  name = 'adk'

  def surface(self):
    return ADK_SURFACE


class SearchAdapter:
  """Implements harness2.core.spec.BenchmarkAdapter for web-search questions."""

  def __init__(
      self,
      *,
      loop_dir: Path | None = None,
      tasks_file: Path = TASKS_FILE,
      base_harness: Path = BASE_HARNESS,
      solver_model: str | BaseLlm | None = None,
  ):
    self._tasks = {t['id']: t for t in json.loads(Path(tasks_file).read_text())}
    self._base = Path(base_harness)
    self._solver_model = solver_model or default_solver_model()
    model_name = self._solver_model if isinstance(self._solver_model, str) else 'custom'
    self.loop_dir = Path(loop_dir or PROJECT / 'runs' / 'loop' / model_name)
    self.executions_dir = self.loop_dir / 'executions'
    self.runs_dir = self.loop_dir / 'runs'
    self.audit_dir = self.loop_dir / 'audit'
    channel = ChannelSpec('claude_cli', improver_model(), effort='high', timeout=1800)
    self.spec = BenchSpec(
        name='search',
        runs_root=self.loop_dir,
        install_mode='workspace_config',
        prompts=PROMPTS,
        channels={'proposer': channel, 'selector': channel},
        domain_word='topic area',
        allow_subagents=True,
    )
    self.substrate = AdkSubstrate()

  # Tasks

  def tasks(self, split: str = 'all') -> list[str]:
    return [t for t, v in self._tasks.items() if split == 'all' or v.get('split') == split]

  def domain_of(self, task_id: str) -> str:
    return self._tasks[task_id]['domain']

  def task_prompt(self, task_id: str) -> str:
    return self._tasks[task_id]['question']

  def has_reference(self, task_id: str) -> bool:
    return bool(self._tasks[task_id].get('reference'))

  def task_files(self, task_id: str) -> Path | None:
    return None

  def base_harness(self) -> Path:
    return self._base

  def validate_tree(self, tree: dict[str, str]) -> None:
    if not any(rel.startswith('sub_agents/') and rel.endswith('/AGENT.md') for rel in tree):
      raise ValueError(
          'the harness has no sub_agents/<name>/AGENT.md, so the agent cannot search the'
          ' web at all; keep sub_agents/searcher/AGENT.md (edit it instead of deleting it)'
      )

  # Running the agent

  def execute(
      self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
  ) -> Path:
    """One rollout, reused when the same harness already completed this run id.

    The harness is copied to a temporary folder first, because scripts and hooks
    the editor wrote run from there.
    """
    source = Path(harness_dir) if harness_dir else self._base
    if not harness_tree.is_tree(ADK_SURFACE, source):
      raise ValueError(f'{source} is not an ADK harness tree (no always-on file present)')
    fingerprint = runmark.fingerprint(source)
    run_dir = self.executions_dir / run_id
    meta = run_dir / 'meta.json'
    if meta.is_file():
      done = json.loads(meta.read_text())
      if done.get('complete') and done.get('harness_fingerprint') == fingerprint:
        return run_dir
    staged = Path(tempfile.mkdtemp(prefix='harness-loop-'))
    try:
      shutil.copytree(source, staged, dirs_exist_ok=True)
      return rollout(
          self.task_prompt(task_id),
          staged,
          run_dir,
          run_id=run_id,
          fingerprint=fingerprint,
          timeout=timeout or ROLLOUT_TIMEOUT_S,
          model=self._solver_model,
      )
    finally:
      shutil.rmtree(staged, ignore_errors=True)

  # What the loop reads back

  def delivered(self, run_dir: Path) -> bool:
    return bool(self.deliverable_text(run_dir).strip())

  def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
    return render_events(
        Path(run_dir) / 'events.jsonl', obs_trunc=obs_trunc, max_chars=TRAJECTORY_MAX_CHARS
    )

  def deliverable_files(self, run_dir: Path) -> dict[str, str]:
    text = self.deliverable_text(run_dir)
    return {'answer.md': text} if text.strip() else {}

  def deliverable_text(self, run_dir: Path) -> str:
    p = Path(run_dir) / 'answer.md'
    return p.read_text(encoding='utf-8', errors='replace') if p.is_file() else ''

  # Grading, after selection only

  def score(self, run_dir: Path, task_id: str) -> dict | None:
    """Gemini compares the answer with the reference; None when there is none."""
    reference = self._tasks[task_id].get('reference')
    answer = self.deliverable_text(run_dir)
    if not reference:
      return None
    if not answer.strip():
      return {'correct': False, 'reason': 'no answer produced'}
    from google import genai

    try:
      # Keep the client in a variable: an unreferenced one is closed before the call.
      client = genai.Client()
      reply = client.models.generate_content(
          model=os.environ.get('HARNESS2_JUDGE_MODEL') or 'gemini-2.5-flash',
          contents=(
              f'Question: {self.task_prompt(task_id)}\n\nReference answer: {reference}\n\n'
              f'Candidate answer:\n{answer}\n\nDoes the candidate state the facts in the'
              ' reference without contradicting them? Small wording or rounding differences'
              ' are fine.'
          ),
          config=types.GenerateContentConfig(
              response_mime_type='application/json',
              response_schema={
                  'type': 'object',
                  'properties': {
                      'correct': {'type': 'boolean'},
                      'reason': {'type': 'string'},
                  },
                  'required': ['correct', 'reason'],
              },
          ),
      )
      return json.loads(reply.text)
    except Exception:
      return None

  def preflight(self) -> None:
    """Checks model credentials are configured; calls no model."""
    vertex = os.environ.get(
        'GOOGLE_GENAI_USE_ENTERPRISE', os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '')
    ).upper() in ('TRUE', '1')
    if vertex and not os.environ.get('GOOGLE_CLOUD_PROJECT') and not os.environ.get(
        'GOOGLE_API_KEY'
    ):
      raise RuntimeError(
          'Vertex AI is selected but GOOGLE_CLOUD_PROJECT is unset; set it in .env'
      )
    if not vertex and not os.environ.get('GOOGLE_API_KEY'):
      raise RuntimeError(
          'GOOGLE_API_KEY is unset; copy .env.example to .env and add a Gemini API key'
      )
    if not shutil.which('claude'):
      raise RuntimeError('the `claude` CLI is not on PATH; the editor runs Claude Code sessions')


def _record(event) -> list[dict]:
  """Flattens one ADK event into trajectory records."""
  out = []
  if event.error_message:
    out.append({'type': 'error', 'author': event.author, 'text': event.error_message})
  for part in (event.content.parts if event.content else None) or []:
    if part.function_call:
      out.append({
          'type': 'tool_call',
          'author': event.author,
          'name': part.function_call.name,
          'args': dict(part.function_call.args or {}),
      })
    elif part.function_response:
      out.append({
          'type': 'tool_result',
          'author': event.author,
          'name': part.function_response.name,
          'response': part.function_response.response,
      })
    elif part.text and part.text.strip():
      kind = 'thinking' if part.thought else 'text'
      out.append({'type': kind, 'author': event.author, 'text': part.text})
  return out


async def _run(agent, question: str) -> tuple[list[dict], str]:
  app_name = f'harness_loop_{uuid.uuid4().hex[:8]}'
  runner = InMemoryRunner(agent=agent, app_name=app_name)
  session = await runner.session_service.create_session(app_name=app_name, user_id='solver')
  message = types.Content(role='user', parts=[types.Part(text=question)])
  records: list[dict] = []
  answer = ''
  async for event in runner.run_async(
      user_id='solver', session_id=session.id, new_message=message
  ):
    records.extend(_record(event))
    if event.author == agent.name and event.is_final_response() and event.content:
      text = ''.join(p.text or '' for p in event.content.parts or [] if not p.thought)
      if text.strip():
        answer = text
  return records, answer


def rollout(
    question: str,
    harness_dir: Path,
    run_dir: Path,
    *,
    run_id: str,
    fingerprint: str,
    timeout: int,
    model: str | BaseLlm,
) -> Path:
  """One ADK execution of `question` under `harness_dir`, written to `run_dir`."""
  run_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  error = ''
  records: list[dict] = []
  answer = ''
  try:
    agent = build_agent(harness_dir, model=model, workdir=harness_dir)
    records, answer = asyncio.run(asyncio.wait_for(_run(agent, question), timeout))
  except TimeoutError:
    error = f'timeout after {timeout}s'
  except Exception as exc:  # A dead rollout is evidence, never a crash of the loop.
    error = f'{type(exc).__name__}: {exc}'
  if error:
    records.append({'type': 'error', 'author': 'substrate', 'text': error})

  with open(run_dir / 'events.jsonl', 'w', encoding='utf-8') as f:
    for r in records:
      f.write(json.dumps(r, default=str, ensure_ascii=False) + '\n')
  (run_dir / 'answer.md').write_text(answer, encoding='utf-8')
  (run_dir / 'meta.json').write_text(
      json.dumps(
          {
              'run_id': run_id,
              'harness_fingerprint': fingerprint,
              'model': model if isinstance(model, str) else type(model).__name__,
              'wall_s': round(time.time() - t0, 1),
              'complete': not error,
              'error': error,
          },
          indent=2,
      ),
      encoding='utf-8',
  )
  return run_dir


def _clip(value, n: int) -> str:
  s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
  return s if n <= 0 or len(s) <= n else s[:n] + f'…(+{len(s) - n} chars — render truncation)'


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400) -> str:
  """events.jsonl in Harness²'s shared [ACTION]/[OBSERVATION]/[ERROR] grammar."""
  tags = {'text': 'ASSISTANT', 'thinking': 'THINKING'}
  lines = []
  try:
    raw = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
  except OSError as exc:
    return f'(failed to read trajectory: {exc})'
  for ln in raw:
    try:
      r = json.loads(ln)
    except json.JSONDecodeError:
      continue
    kind = r.get('type')
    if kind in tags:
      lines.append(f"[{tags[kind]}] {_clip(r.get('text', ''), obs_trunc)}")
    elif kind == 'tool_call':
      lines.append(f"[ACTION] {r.get('name')}({_clip(r.get('args', {}), args_trunc)})")
    elif kind == 'tool_result':
      response = r.get('response')
      is_error = isinstance(response, dict) and set(response) == {'error'}
      tag = 'OBSERVATION ERROR' if is_error else 'OBSERVATION'
      lines.append(f'[{tag}] {_clip(response, obs_trunc)}')
    elif kind == 'error':
      lines.append(f"[ERROR] {_clip(r.get('text', ''), 300)}")
  rendered = '\n'.join(lines).strip()
  if not rendered:
    return '(no parseable events in trajectory)'
  if len(rendered) > max_chars:
    rendered = rendered[:max_chars] + f'\n…(+{len(rendered) - max_chars} chars — render truncation)'
  return rendered
