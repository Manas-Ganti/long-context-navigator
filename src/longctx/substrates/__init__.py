"""Substrates: ways of producing `Instance` objects for the same environment.

`synthetic` (longctx.generate) builds corpora whose every property is known by
construction. `musique` lays real Wikipedia paragraphs from a multi-hop QA
dataset into the same chunk format, so the same environment, reward, audit and
trainers run unchanged on real text.

The trade is explicit and reported: the synthetic substrate can PROVE each
instance is solvable and that no distractor path reaches the answer (a parser
re-derives the answer from the text); a real substrate relies on the dataset's
own annotations instead.
"""

from . import musique  # noqa: F401

__all__ = ["musique"]
