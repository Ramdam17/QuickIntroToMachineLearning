"""Kind automatic checks for the exercises of the course notebooks.

An exercise is a code cell the learner completes. The cell after it calls one of the two functions
below, which compares the learner's answer with the expected value and prints one message:

- :func:`equal` checks a value the learner computed.
- :func:`returns` checks a function the learner wrote, by calling it.

A check never raises in a notebook. A cell left unfilled (its ``...`` placeholder still in place)
reports "Not filled in yet", so a notebook executes from top to bottom whatever the learner has or
has not written. The messages go through ``print`` on purpose: they are the visible answer of a
notebook cell to the learner, not a log.

The course's own guard (``scripts/check_exercises.py``) needs a failed check to stop the execution.
It sets the environment variable named by :data:`STRICT_ENV`; a check that does not pass then
raises :class:`AssertionError`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["DEFAULT_TOLERANCE", "STRICT_ENV", "equal", "returns"]

# Environment variable that turns a check that does not pass into an AssertionError.
STRICT_ENV = "ML_COURSE_STRICT_CHECKS"
# Largest gap accepted between a numeric answer and the expected value, used both as an absolute
# and as a relative tolerance. It absorbs floating-point rounding (two correct ways of ordering a
# sum differ by about 1e-15) while rejecting any real mistake. A practical choice, not a standard.
DEFAULT_TOLERANCE = 1e-8

_NOT_FILLED = "Not filled in yet. Replace the `...` with your code, then run this cell again."
_CORRECT = "Correct. Your answer matches the expected value."


def _is_unfilled(value: Any) -> bool:
    """Return True when the value is still the placeholder of a cell to complete."""
    if value is ... or value is None:
        return True
    if isinstance(value, list | tuple):
        return any(item is ... for item in value)
    return False


def _describe(value: Any) -> str:
    """Return a short text for a value, safe to print in a message."""
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _compare(answer: Any, expected: Any, tol: float) -> tuple[bool, str]:
    """Compare an answer with the expected value.

    Returns
    -------
    tuple of (bool, str)
        Whether the answer matches, and when it does not, a sentence saying what differs.
    """
    # A boolean or a text is compared exactly: "close to True" has no meaning.
    if isinstance(expected, bool | np.bool_ | str):
        same_kind = isinstance(answer, type(expected)) or (
            isinstance(expected, bool | np.bool_) and isinstance(answer, bool | np.bool_)
        )
        if same_kind and bool(answer == expected):
            return True, ""
        return False, f"Your answer is {_describe(answer)}; the expected value is {expected!r}."

    try:
        answer_array = np.asarray(answer)
        expected_array = np.asarray(expected)
    except Exception:  # noqa: BLE001 - an answer that numpy cannot read is compared as is
        matches = bool(answer == expected)
        return matches, "" if matches else "Your answer does not match the expected value."

    if answer_array.shape != expected_array.shape:
        if expected_array.shape == ():
            return False, f"A single value is expected; your answer is {_describe(answer)}."
        return (
            False,
            f"Your answer has shape {answer_array.shape}; the expected shape is "
            f"{expected_array.shape}.",
        )

    numeric = np.issubdtype(answer_array.dtype, np.number) and np.issubdtype(
        expected_array.dtype, np.number
    )
    if numeric:
        matches = bool(np.allclose(answer_array, expected_array, rtol=tol, atol=tol))
    else:
        matches = bool(np.array_equal(answer_array, expected_array))
    if matches:
        return True, ""
    if expected_array.shape == ():
        return False, f"Your answer is {_describe(answer)}; the expected value is {expected!r}."
    return False, "The shape is right, but at least one value differs from the expected one."


def _report(label: str, passed: bool, message: str, hint: str | None = None) -> None:
    """Print the outcome of a check, and raise instead when strict checks are requested."""
    if not passed and hint:
        message = f"{message} {hint}"
    if not passed and os.environ.get(STRICT_ENV):
        raise AssertionError(f"{label}: {message}")
    print(f"{label}: {message}")


def equal(
    label: str,
    answer: Any,
    expected: Any,
    *,
    tol: float = DEFAULT_TOLERANCE,
    hint: str | None = None,
) -> None:
    """Check a value the learner computed against the expected value.

    When to use: in the cell that follows a cell to complete whose result is a value (a number, an
    array, a Series, a text, a boolean). For a function the learner wrote, use :func:`returns`.

    Parameters
    ----------
    label : str
        Name printed in front of the message, for example ``"Exercise 1"``.
    answer : Any
        The learner's value. ``...`` and ``None`` mean the cell was not filled in.
    expected : Any
        The expected value. A number, or an array-like of any shape (list, ``numpy.ndarray``,
        ``pandas.Series``, ``pandas.DataFrame``), is compared within ``tol``; the shapes must be
        the same. A text or a boolean is compared exactly.
    tol : float, default 1e-8
        Absolute and relative tolerance for numeric values, in the unit of the value.
    hint : str, optional
        A sentence added to the message when the answer does not match, pointing the learner to
        the part of the notebook to read again.

    Returns
    -------
    None
        The outcome is printed. Nothing is raised, unless the environment variable named by
        :data:`STRICT_ENV` is set, in which case a check that does not pass raises
        ``AssertionError``.

    Examples
    --------
    >>> from ml_course import check
    >>> check.equal("Exercise 1", 0.5 * 2.0 + 1.5 * -1.0 + 0.25, expected=-0.25)
    Exercise 1: Correct. Your answer matches the expected value.
    >>> check.equal("Exercise 1", ..., expected=-0.25)
    Exercise 1: Not filled in yet. Replace the `...` with your code, then run this cell again.
    """
    if _is_unfilled(answer):
        _report(label, False, _NOT_FILLED)
        return
    try:
        matches, difference = _compare(answer, expected, tol)
    except Exception as exc:  # noqa: BLE001 - a check must never break the learner's notebook
        matches = False
        difference = f"Your answer could not be compared ({type(exc).__name__}: {exc})."
    if matches:
        _report(label, True, _CORRECT)
    else:
        _report(label, False, f"Not yet. {difference}", hint)


def returns(
    label: str,
    function: Callable[..., Any],
    *args: Any,
    expected: Any,
    tol: float = DEFAULT_TOLERANCE,
    hint: str | None = None,
    **kwargs: Any,
) -> None:
    """Check a function the learner wrote, by calling it and comparing what it returns.

    When to use: in the cell that follows a cell to complete that defines a function. The call
    happens inside the check, so a function that is unfilled or that raises is reported kindly
    instead of stopping the notebook.

    Parameters
    ----------
    label : str
        Name printed in front of the message, for example ``"Exercise 2"``.
    function : callable
        The learner's function.
    *args, **kwargs : Any
        The arguments the function is called with.
    expected : Any
        The value the function must return for these arguments, compared as in :func:`equal`.
    tol : float, default 1e-8
        Absolute and relative tolerance for numeric values, in the unit of the value.
    hint : str, optional
        A sentence added to the message when the answer does not match.

    Returns
    -------
    None
        The outcome is printed, with the same strict mode as :func:`equal`.

    Examples
    --------
    >>> from ml_course import check
    >>> def neuron_score(x, w, b):
    ...     return w * x + b
    >>> check.returns("Exercise 2", neuron_score, 2.0, 0.5, b=0.25, expected=1.25)
    Exercise 2: Correct. Your answer matches the expected value.
    """
    if not callable(function):
        _report(label, False, _NOT_FILLED)
        return
    try:
        answer = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the learner's error is reported, not raised
        _report(
            label,
            False,
            f"Not yet. Your function stopped with {type(exc).__name__}: {exc}.",
            hint,
        )
        return
    equal(label, answer, expected, tol=tol, hint=hint)
