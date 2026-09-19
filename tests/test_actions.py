import pytest

from longctx.actions import ANSWER, COMPRESS, DROP, READ, ParseError, parse_action


def test_read_with_thought():
    a = parse_action("THOUGHT: look.\nACTION: READ 12")
    assert a.kind == READ and a.ids == ["12"]


def test_last_action_line_wins():
    a = parse_action("ACTION: READ 1\nACTION: DROP 1")
    assert a.kind == DROP


def test_compress_ids_and_summary():
    a = parse_action("ACTION: COMPRESS [12, S1] :: Amber Falcon: Lead = Dana Ruiz")
    assert a.kind == COMPRESS and a.ids == ["12", "S1"] and a.text == "Amber Falcon: Lead = Dana Ruiz"


def test_drop_space_separated():
    assert parse_action("DROP 3 4 s2").ids == ["3", "4", "S2"]


def test_answer_keeps_text():
    a = parse_action("ACTION: ANSWER 4,885,866")
    assert a.kind == ANSWER and a.text == "4,885,866"


@pytest.mark.parametrize("bad", ["", "ACTION: READ", "READ S1", "READ 1 2", "COMPRESS 1", "COMPRESS 1 ::  ",
                                 "ANSWER", "JUMP 3", "DROP", "READ x"])
def test_invalid(bad):
    with pytest.raises(ParseError):
        parse_action(bad)
