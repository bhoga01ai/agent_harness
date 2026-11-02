"""improve_harness: the one command that runs the harness loop.

    uv run improve-harness --k 1 --mode parallel

For every task, Harness² `run_domain()` proposes edits to copies of the
harness, probes them with agent rollouts, composes, selects and remembers.
Everything is written to runs/loop/<model>/:

    executions/   every agent answer
    runs/         edited harnesses and the selection decisions
    audit/        every editor session

It ends with a scoreboard (base vs adopted, graded against the reference
answers) and the best harness, in best_harness/. --adopt copies that harness
into harness/active, which `adk web` serves, but only if it scored higher.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness2 import config
from harness2.core import schedule
from harness2.core import tree as harness_tree

from search_agent.agent import ADK_SURFACE, build_agent

from .adapter import PROJECT, SearchAdapter
from .proposer import LocalClaudeProposer

ACTIVE_HARNESS = PROJECT / 'harness' / 'active'
HISTORY = PROJECT / 'harness' / 'history'


def check_harness(adapter: SearchAdapter) -> bool:
  """Validates the base harness and builds the agent from it; calls no model."""
  base = adapter.base_harness()
  tree = harness_tree.read_tree(ADK_SURFACE, base)
  try:
    census = harness_tree.validate(ADK_SURFACE, tree, allow_subagents=True)
    adapter.validate_tree(tree)
    agent = build_agent(base, model='validation-only')
  except ValueError as exc:
    print(f'Harness {base} is invalid:')
    for problem in str(exc).split(' | '):
      print(f'  - {problem}')
    return False
  tools = [getattr(t, 'name', None) or t.__name__ for t in agent.tools]
  print(
      f"Harness ok: {census['n_files']} files, {census['always_on_chars']} always-on chars,"
      f" {census['sub_agents']} sub-agent(s), {census['skills']} skill(s),"
      f" {census['scripts']} script(s), {'a' if census['hooks'] else 'no'} hooks file;"
      f" agent tools: {', '.join(tools)}"
  )
  return True


def _correct(adapter: SearchAdapter, run_dir: str, task_id: str) -> int | None:
  result = adapter.score(Path(run_dir), task_id)
  return None if result is None else int(bool(result.get('correct')))


def scoreboard(adapter: SearchAdapter, task_ids: list[str], k: int, mode: str) -> list[dict]:
  """One row per task: base and adopted grades, and the harness the loop selected."""
  bench, sub = adapter.spec.name, adapter.substrate.name
  rows = []
  for tid in task_ids:
    summary = schedule.Namespace(bench, sub, 'agg', k).task_dir(adapter.runs_dir, tid)
    summary = summary / 'summary.json'
    if not summary.is_file():
      continue
    s = json.loads(summary.read_text())
    sel = (s.get('selections') or {}).get(mode) or {}
    base = [_correct(adapter, b['run_dir'], tid) for b in s.get('base') or []]
    base = [b for b in base if b is not None]
    adopted = _correct(adapter, sel['chosen_run_dir'], tid) if sel.get('chosen_run_dir') else None
    harness = schedule.Namespace(bench, sub, mode, k).task_dir(adapter.runs_dir, tid)
    harness = harness / 'harnesses' / str(sel.get('chosen_label'))
    rows.append({
        'task': tid,
        'base': sum(base) / len(base) if base else None,
        'adopted': adopted,
        'harness': str(harness),
        'edited': bool(sel.get('chosen_label')) and not sel.get('chosen_tree_is_base'),
        'why': str(sel.get('why') or '').strip(),
    })
  return rows


def best_harness(rows: list[dict]) -> dict | None:
  """The edited harness that gained most over base; ties go to the later task."""
  edited = [
      (i, r) for i, r in enumerate(rows)
      if r['edited'] and harness_tree.is_tree(ADK_SURFACE, Path(r['harness']))
  ]
  if not edited:
    return None
  graded = [(i, r) for i, r in edited if r['base'] is not None and r['adopted'] is not None]
  if not graded:
    return edited[-1][1]
  return max(graded, key=lambda ir: (ir[1]['adopted'] - ir[1]['base'], ir[1]['adopted'], ir[0]))[1]


def scored_higher(row: dict) -> bool:
  return row['base'] is not None and row['adopted'] is not None and row['adopted'] > row['base']


def adopt(harness_dir: Path) -> None:
  if ACTIVE_HARNESS.exists():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    HISTORY.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ACTIVE_HARNESS), HISTORY / stamp)
    print(f'previous active harness saved to {HISTORY / stamp}')
  tree = harness_tree.read_tree(ADK_SURFACE, harness_dir)
  harness_tree.write_tree(ADK_SURFACE, tree, ACTIVE_HARNESS)
  print(f'adopted -> {ACTIVE_HARNESS} (restart `adk web` to pick it up)')


def report(adapter: SearchAdapter, rows: list[dict], do_adopt: bool) -> int:
  graded = [r for r in rows if r['base'] is not None and r['adopted'] is not None]
  base = sum(r['base'] for r in graded)
  adopted = sum(r['adopted'] for r in graded)
  print(f'\nScoreboard: base {base:g}/{len(graded)} -> adopted {adopted:g}/{len(graded)}')
  for r in rows:
    grades = (
        'no reference' if r['base'] is None or r['adopted'] is None
        else f"base {r['base']:g} -> adopted {r['adopted']}"
    )
    print(f"  {r['task']}: {grades}{'' if r['edited'] else ' (kept base harness)'}")

  best = best_harness(rows)
  (adapter.loop_dir / 'scoreboard.json').write_text(
      json.dumps({'rows': rows, 'best': best}, indent=2)
  )
  if best is None:
    print('\nNo edited harness was selected; nothing to adopt.')
    return 0
  out = adapter.loop_dir / 'best_harness'
  shutil.rmtree(out, ignore_errors=True)
  harness_tree.write_tree(ADK_SURFACE, harness_tree.read_tree(ADK_SURFACE, Path(best['harness'])), out)
  print(f"\nBest harness (from {best['task']}): {out}")
  if best['why']:
    print(f"  why: {best['why'][:400]}")
  if not scored_higher(best):
    print('  It did not score higher than base, so it is not adopted.')
  elif do_adopt:
    adopt(out)
  else:
    print('  It scored higher than base. Re-run with --adopt to serve it in `adk web`.')
  return 0


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog='improve-harness', description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--k', type=int, default=1,
                 help='scale: about 4k rollouts and 3k editor sessions per task')
  p.add_argument('--mode', default='parallel', help='parallel | sequential')
  p.add_argument('--only-tasks', default='', help='comma list of task ids')
  p.add_argument('--max-tasks', type=int, default=None, help='per topic area')
  p.add_argument('--dry-run', action='store_true',
                 help='validate the harness, print the tasks and cost, and stop')
  p.add_argument('--adopt', action='store_true',
                 help='copy the best harness into harness/active if it scored higher')
  args = p.parse_args(argv)

  config.load_dotenv(PROJECT / '.env')
  adapter = SearchAdapter()
  if not check_harness(adapter):
    return 1
  mode = schedule.mode_token(args.mode)
  stream = schedule.task_stream(
      adapter,
      only_tasks=tuple(t for t in args.only_tasks.split(',') if t),
      max_tasks=args.max_tasks,
  )
  task_ids = [t for tasks in stream.values() for t in tasks]
  print(
      f'{len(task_ids)} task(s), k={args.k}, {schedule.MODE_NAMES[mode]} mode: about'
      f' {4 * args.k * len(task_ids)} agent rollouts and'
      f' {3 * args.k * len(task_ids)} editor sessions -> {adapter.loop_dir}'
  )
  for domain, tasks in stream.items():
    print(f'  {domain}: {", ".join(tasks)}')
  if args.dry_run or not task_ids:
    return 0
  adapter.preflight()

  prop = LocalClaudeProposer(adapter.spec.channels['proposer'], audit_dir=adapter.audit_dir)
  failed = 0
  for domain, tasks in stream.items():
    out = schedule.run_domain(
        domain, tasks, args.k,
        adapter=adapter,
        prop=prop,
        runs_root=adapter.runs_dir,
        library_root=adapter.runs_dir / '_library',
        modes=(mode,),
    )
    failed += out['failed']
  report(adapter, scoreboard(adapter, task_ids, args.k, mode), args.adopt)
  return 2 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
