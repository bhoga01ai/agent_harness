"""Days, and whole years and months, between two ISO dates.

Usage: python scripts/date_diff.py 2022-12-18 [2026-11-01]
The second date defaults to today. Prints JSON.
"""

import datetime
import json
import sys


def main(argv: list[str]) -> int:
  if not 1 <= len(argv) <= 2:
    print(__doc__, file=sys.stderr)
    return 2
  try:
    start = datetime.date.fromisoformat(argv[0])
    end = datetime.date.fromisoformat(argv[1]) if len(argv) == 2 else datetime.date.today()
  except ValueError as exc:
    print(f'expected YYYY-MM-DD dates: {exc}', file=sys.stderr)
    return 2
  sign = 1 if end >= start else -1
  a, b = (start, end) if sign > 0 else (end, start)
  months = (b.year - a.year) * 12 + b.month - a.month - (b.day < a.day)
  print(json.dumps({
      'days': sign * (b - a).days,
      'years': sign * (months // 12),
      'months': sign * (months % 12),
  }))
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
