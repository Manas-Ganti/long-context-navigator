"""Action grammar and parser.

    READ <chunk id>
    COMPRESS <ids> :: <summary text>
    DROP <ids>
    ANSWER <text>

ids are chunk indices (integers) or summary ids (S1, S2, ...), separated by
commas or spaces, optionally in brackets. The parser takes the LAST line that
starts with 'ACTION:' together with any lines after it (so a summary may run
on), or the last non-empty line if no such prefix exists — a policy may write
a THOUGHT line first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

READ, COMPRESS, DROP, ANSWER = "READ", "COMPRESS", "DROP", "ANSWER"
KINDS = (READ, COMPRESS, DROP, ANSWER)

_ID = re.compile(r"^(?:S\d+|\d+)$", re.IGNORECASE)


@dataclass
class Action:
    kind: str
    ids: list[str] = field(default_factory=list)   # normalised: "12" or "S1"
    text: str | None = None

    def __str__(self) -> str:
        if self.kind == READ:
            return f"READ {self.ids[0]}"
        if self.kind == DROP:
            return f"DROP {', '.join(self.ids)}"
        if self.kind == COMPRESS:
            return f"COMPRESS {', '.join(self.ids)} :: {self.text}"
        return f"ANSWER {self.text}"


class ParseError(ValueError):
    pass


def _parse_ids(s: str) -> list[str]:
    s = s.strip().strip("[]()")
    parts = [p for p in re.split(r"[,\s]+", s) if p]
    if not parts:
        raise ParseError("no ids given")
    out = []
    for p in parts:
        if not _ID.match(p):
            raise ParseError(f"bad id {p!r}")
        out.append(p.upper() if p[0] in "sS" else str(int(p)))
    return out


def parse_action(text: str) -> Action:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        raise ParseError("empty response")
    tagged = [i for i, l in enumerate(lines) if l.upper().startswith("ACTION:")]
    if tagged:
        # the action line plus anything after it (a summary may run on)
        i = tagged[-1]
        line = " ".join([lines[i][len("ACTION:"):].strip()] + lines[i + 1:]).strip()
    else:
        line = lines[-1]
    m = re.match(r"^(READ|COMPRESS|DROP|ANSWER)\b(.*)$", line, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ParseError(f"unrecognised action line {line!r}")
    kind, rest = m.group(1).upper(), m.group(2).strip()
    if kind == READ:
        ids = _parse_ids(rest)
        if len(ids) != 1 or ids[0].startswith("S"):
            raise ParseError("READ takes exactly one chunk id")
        return Action(READ, ids)
    if kind == DROP:
        return Action(DROP, _parse_ids(rest))
    if kind == COMPRESS:
        if "::" not in rest:
            raise ParseError("COMPRESS needs '<ids> :: <summary>'")
        ids_part, summary = rest.split("::", 1)
        summary = summary.strip()
        if not summary:
            raise ParseError("COMPRESS summary is empty")
        return Action(COMPRESS, _parse_ids(ids_part), summary)
    if not rest:
        raise ParseError("ANSWER is empty")
    return Action(ANSWER, [], rest)
