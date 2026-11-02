"""PostToolUse hook: flags a searcher result that carries no source URL.

Reads the hook event as JSON on stdin. Prints {"decision": "block", "reason": ...}
when the result has no http(s) link, which the agent sees as `hook_feedback`.
"""

import json
import sys

event = json.load(sys.stdin)
response = json.dumps(event.get('tool_response'), ensure_ascii=False)
if 'http://' not in response and 'https://' not in response:
  print(json.dumps({
      'decision': 'block',
      'reason': (
          'This searcher result has no source URLs. Ask the searcher again for the same'
          ' facts with the title and URL of each source before using them.'
      ),
  }))
else:
  print('{}')
