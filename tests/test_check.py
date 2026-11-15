"""Tests for the exercise checks: right, wrong and unfilled answers, and the strict mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_course import check

CORRECT = "Correct. Your answer matches the expected value."
NOT_FILLED = "Not filled in yet."


@pytest.fixture(autouse=True)
def _no_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts in the learner's mode, whatever the environment of the test run."""
    monkeypatch.delenv(check.STRICT_ENV, raising=False)


def _message(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (-0.25, -0.25),
        (0.1 + 0.2, 0.3),  # floating-point rounding must not fail a correct answer
        (3, 3.0),
        ([1.0, 2.0], np.array([1.0, 2.0])),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        (pd.Series([0.5, 1.5]), [0.5, 1.5]),
        (pd.DataFrame({"a": [1, 2]}), [[1], [2]]),
        ("Adelie", "Adelie"),
        (True, True),
        (np.bool_(False), False),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_equal_accepts_a_correct_answer(
    capsys: pytest.CaptureFixture[str], answer: object, expected: object
) -> None:
    result = check.equal("Exercise 1", answer, expected=expected)

    assert result is None
    assert _message(capsys) == f"Exercise 1: {CORRECT}"


@pytest.mark.parametrize(
    ("answer", "expected", "fragment"),
    [
        (-0.5, -0.25, "Your answer is -0.5; the expected value is -0.25."),
        (0.3001, 0.3, "the expected value is 0.3"),
        ([1.0, 2.0, 3.0], [1.0, 2.0], "shape (3,); the expected shape is (2,)"),
        ([1.0, 2.5], [1.0, 2.0], "at least one value differs"),
        ([1.0, 2.0], 3.0, "A single value is expected"),
        ("adelie", "Adelie", "the expected value is 'Adelie'"),
        (1, True, "the expected value is True"),
        ("0.3", 0.3, "the expected value is 0.3"),
    ],
)
def test_equal_rejects_a_wrong_answer_kindly(
    capsys: pytest.CaptureFixture[str], answer: object, expected: object, fragment: str
) -> None:
    check.equal("Exercise 1", answer, expected=expected)

    message = _message(capsys)
    assert message.startswith("Exercise 1: Not yet.")
    assert fragment in message


@pytest.mark.parametrize("answer", [..., None, [..., 1.0], (2.0, ...)])
def test_equal_reports_an_unfilled_cell_without_raising(
    capsys: pytest.CaptureFixture[str], answer: object
) -> None:
    check.equal("Exercise 1", answer, expected=1.0)

    assert _message(capsys).startswith(f"Exercise 1: {NOT_FILLED}")


def test_equal_appends_the_hint_only_when_wrong(capsys: pytest.CaptureFixture[str]) -> None:
    hint = "Read the worked example again."

    check.equal("Exercise 1", 2.0, expected=1.0, hint=hint)
    assert _message(capsys).endswith(hint)

    check.equal("Exercise 1", 1.0, expected=1.0, hint=hint)
    assert hint not in _message(capsys)


def test_equal_respects_the_tolerance(capsys: pytest.CaptureFixture[str]) -> None:
    check.equal("Exercise 1", 0.731, expected=0.7310585786, tol=1e-3)
    assert _message(capsys).endswith(CORRECT)

    check.equal("Exercise 1", 0.731, expected=0.7310585786)
    assert "Not yet." in _message(capsys)


def test_equal_never_raises_on_an_answer_it_cannot_compare(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check.equal("Exercise 1", {"a": 1}, expected=[1.0, 2.0])

    assert "Not yet." in _message(capsys)


def test_returns_calls_the_function(capsys: pytest.CaptureFixture[str]) -> None:
    def neuron_score(x: float, w: float, b: float) -> float:
        return w * x + b

    check.returns("Exercise 2", neuron_score, 2.0, 0.5, b=0.25, expected=1.25)

    assert _message(capsys) == f"Exercise 2: {CORRECT}"


def test_returns_reports_an_unfilled_function(capsys: pytest.CaptureFixture[str]) -> None:
    def neuron_score(x: float, w: float, b: float) -> object:
        return ...  # the placeholder of a cell to complete

    check.returns("Exercise 2", neuron_score, 2.0, 0.5, 0.25, expected=1.25)
    assert _message(capsys).startswith(f"Exercise 2: {NOT_FILLED}")

    check.returns("Exercise 2", ..., 2.0, expected=1.25)
    assert _message(capsys).startswith(f"Exercise 2: {NOT_FILLED}")


def test_returns_reports_an_error_in_the_function(capsys: pytest.CaptureFixture[str]) -> None:
    def broken(x: float) -> float:
        return x / 0

    check.returns("Exercise 2", broken, 1.0, expected=1.0)

    message = _message(capsys)
    assert message.startswith("Exercise 2: Not yet. Your function stopped with ZeroDivisionError")


@pytest.mark.parametrize("answer", [..., 2.0])
def test_strict_mode_raises_on_anything_but_a_correct_answer(
    monkeypatch: pytest.MonkeyPatch, answer: object
) -> None:
    monkeypatch.setenv(check.STRICT_ENV, "1")

    with pytest.raises(AssertionError, match="Exercise 1"):
        check.equal("Exercise 1", answer, expected=1.0)


def test_strict_mode_lets_a_correct_answer_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(check.STRICT_ENV, "1")

    check.equal("Exercise 1", 1.0, expected=1.0)

    assert _message(capsys).endswith(CORRECT)


def test_messages_avoid_the_words_the_course_bans() -> None:
    banned = ("obviously", "simply", "trivially", "just")
    for text in (check._NOT_FILLED, check._CORRECT):
        assert not any(word in text.lower() for word in banned)
