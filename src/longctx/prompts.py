"""Prompt rendering. The policy is STATELESS between steps: every step it sees
exactly this — the question, the document map, the memory ledger and the
working context — and nothing of its previous reasoning. A summary written by
COMPRESS is therefore the only way to carry a bridge entity across reads."""

from __future__ import annotations

SYSTEM_PROMPT = """You are navigating a document that is far larger than the context you may hold.
Answer the question by reading chunks, keeping only what you need, and answering.

Rules of the working context:
- The working context holds chunks you READ and summaries you COMPRESS. Its size is
  measured in tokens against a HARD ceiling. If any action would push it above the
  ceiling, the episode ends immediately as a failure. Plan around chunk sizes shown
  in the document map.
- You do NOT remember anything between steps except what is in the working context.
  If you need a name or value later, write it into a COMPRESS summary before
  freeing the chunk that contains it.
- Every action costs one step from a fixed budget. Re-reading a dropped chunk is
  allowed but costs a step.
- Entries can carry superseded values ("previously X", "then Y [current ...]").
  Use the CURRENT value. Entities with similar names are different entities.
- Registers are sorted alphabetically and the map gives each chunk's range. To
  locate an entry, compare its name against those ranges before reading — a
  chunk that does not contain it costs a step for nothing.

Actions (exactly one per response, on the last line, prefixed with ACTION:):
  ACTION: READ <chunk id>                       load a chunk
  ACTION: COMPRESS <ids> :: <summary>           replace held items with your own summary
  ACTION: DROP <ids>                            remove held items, freeing their tokens
  ACTION: ANSWER <value>                        final answer (the value only)

Write one short THOUGHT line, then the ACTION line."""


def render_map(obs) -> str:
    return "\n".join(obs.map_lines)


def render_observation(obs) -> str:
    free = obs.ceiling - obs.used
    lines = [
        f"QUESTION: {obs.question}",
        "",
        f"CONTEXT: {obs.used} / {obs.ceiling} tokens used ({free} free). Exceeding {obs.ceiling} ends the episode.",
        f"STEPS: {obs.steps_used} used of {obs.step_budget}.",
        "",
        "DOCUMENT MAP (id · section · first entry – last entry · tokens):",
        render_map(obs),
        "",
        "MEMORY LEDGER:",
        f"  read so far: {', '.join(map(str, obs.read)) or '—'}",
        f"  dropped: {', '.join(map(str, obs.dropped)) or '—'}",
        f"  compressed: {', '.join(obs.compressed) or '—'}",
        f"  last action: {obs.last_action_result or '—'}",
        "",
        f"WORKING CONTEXT ({obs.used} tokens):",
    ]
    if not obs.held:
        lines.append("  (empty)")
    for item in obs.held:
        if item.kind == "chunk":
            lines.append(f"[{item.id}] ({item.tokens} tokens)")
            lines.append(item.text)
        else:
            lines.append(f"[{item.id}] ({item.tokens} tokens) summary of {', '.join(item.sources)}: {item.text}")
        lines.append("")
    lines.append("Respond with one THOUGHT line and one ACTION line.")
    return "\n".join(lines)


def build_messages(obs) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(obs)}]
