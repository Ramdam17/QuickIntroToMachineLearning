"""Tests for the notebook guards: cell length, cross-references and exercises."""

from __future__ import annotations

from pathlib import Path

import check_cells
import check_crossrefs
import check_exercises
import nbformat
import pytest

from ml_course import check

SOLUTION = (
    "<details>\n<summary>Show the solution</summary>\n\n"
    "```python\n{block}\n```\n\nA word of explanation.\n\n</details>"
)


def _write(path: Path, cells: list[tuple[str, str]]) -> Path:
    """Write a notebook from ``(kind, source)`` pairs, kind being ``"code"`` or ``"md"``."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        (
            nbformat.v4.new_code_cell(source)
            if kind == "code"
            else nbformat.v4.new_markdown_cell(source)
        )
        for kind, source in cells
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


def _exercise(number: int, scaffold: str, check_call: str, solution: str) -> list[tuple[str, str]]:
    """Return the three cells of one exercise."""
    return [
        ("code", f"# Exercise {number}\n{scaffold}"),
        ("code", check_call),
        ("md", SOLUTION.format(block=f"# Exercise {number}\n{solution}")),
    ]


# --- check_cells -------------------------------------------------------------------------------


def test_count_code_lines_skips_blank_and_comment_lines() -> None:
    source = "# a comment\n\nw = 0.5\n    # indented comment\nb = 0.25  # trailing comment\n"

    assert check_cells.count_code_lines(source) == 2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f(x):\n    return x\n", True),
        ("# the model\nclass Model:\n    pass\n", True),
        ("def f(x):\n    return x\n\ny = f(1)\n", False),
        ("x = 1\n", False),
        ("%matplotlib inline\ndef f(x):\n    return x\n", True),
        ("def broken(:\n", False),
    ],
)
def test_is_single_definition(source: str, expected: bool) -> None:
    assert check_cells.is_single_definition(source) is expected


def test_check_cells_fails_on_a_long_cell_but_allows_a_single_definition(tmp_path: Path) -> None:
    long_script = "\n".join(f"x{i} = {i}" for i in range(26))
    long_function = "def f():\n" + "\n".join(f"    x{i} = {i}" for i in range(26))
    at_the_limit = "\n".join(f"x{i} = {i}" for i in range(25)) + "\n# a comment\n\n"
    _write(tmp_path / "01_M" / "01_long.ipynb", [("code", long_script)])
    _write(tmp_path / "01_M" / "02_fine.ipynb", [("code", long_function), ("code", at_the_limit)])

    found = check_cells.long_cells(tmp_path / "01_M" / "02_fine.ipynb")

    assert [(cell.index, cell.code_lines, cell.single_definition) for cell in found] == [
        (0, 27, True)
    ]
    assert check_cells.main(["--notebook-dir", str(tmp_path), "01_M/02_*.ipynb"]) == 0
    assert check_cells.main(["--notebook-dir", str(tmp_path)]) == 1
    assert check_cells.main(["--notebook-dir", str(tmp_path), "nothing/*.ipynb"]) == 2


# --- check_crossrefs ---------------------------------------------------------------------------


def _course(tmp_path: Path, middle_cells: list[tuple[str, str]]) -> Path:
    """Write a three-notebook course; the middle notebook holds the cells under test."""
    _write(
        tmp_path / "00_Start" / "01_first.ipynb",
        [("md", "Previous: none · Next: module 00, notebook 2 (second)")],
    )
    middle = _write(tmp_path / "00_Start" / "02_second.ipynb", middle_cells)
    _write(
        tmp_path / "01_Next" / "01_third.ipynb",
        [("md", "Previous: module 00, notebook 2 (second) · Next: none")],
    )
    return middle


GOOD_LINKS = "Previous: module 00, notebook 1 (first) · Next: module 01, notebook 1 (third)"


def test_crossrefs_pass_on_a_consistent_course(tmp_path: Path) -> None:
    _course(tmp_path, [("md", "As seen in Module 00, notebook 1 (first)."), ("md", GOOD_LINKS)])

    assert check_crossrefs.main(["--notebook-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("See module 07, notebook 3.", "`module 07, notebook 3` does not exist"),
        ("See chapter 3 for the sigmoid.", "`chapter 3` is not in the wording"),
        ("Recall NB 2.", "`NB 2` is not in the wording"),
        ("In notebook 1 we saw it.", "`notebook 1` is not in the wording"),
    ],
)
def test_crossrefs_report_a_bad_reference(tmp_path: Path, text: str, fragment: str) -> None:
    middle = _course(tmp_path, [("md", text), ("md", GOOD_LINKS)])

    problems = check_crossrefs.check_notebook(middle, check_crossrefs.course_order(tmp_path))

    assert len(problems) == 1
    assert fragment in problems[0]


@pytest.mark.parametrize(
    "text",
    [
        "The call `plot(chapter 3)` and\n```python\n# see NB 2\n```\nare code.",
        "Goodfellow et al., chapter 6 of *Deep Learning*.",
    ],
)
def test_crossrefs_ignore_code_and_book_chapters(tmp_path: Path, text: str) -> None:
    middle = _course(tmp_path, [("md", text), ("md", GOOD_LINKS)])

    assert check_crossrefs.check_notebook(middle, check_crossrefs.course_order(tmp_path)) == []


@pytest.mark.parametrize(
    ("links", "fragment"),
    [
        ("Next: module 01, notebook 1", "no `Previous:` entry"),
        ("Previous: module 00, notebook 1", "no `Next:` entry"),
        (
            "Previous: module 01, notebook 1 · Next: module 01, notebook 1",
            "expected `Previous: module 00, notebook 1`",
        ),
        ("Previous: module 00, notebook 1 · Next: none", "expected `Next: module 01, notebook 1`"),
    ],
)
def test_crossrefs_report_wrong_neighbours(tmp_path: Path, links: str, fragment: str) -> None:
    middle = _course(tmp_path, [("md", links)])

    problems = check_crossrefs.check_notebook(middle, check_crossrefs.course_order(tmp_path))

    assert any(fragment in problem for problem in problems)


# --- check_exercises ---------------------------------------------------------------------------

SETUP = ("code", "from ml_course import check\n\nw, x, b = 0.5, 2.0, 0.25")


def test_exercises_pass_when_well_formed_and_solved(tmp_path: Path) -> None:
    notebook = _write(
        tmp_path / "01_M" / "01_good.ipynb",
        [
            SETUP,
            *_exercise(
                1,
                "score = ...  # YOUR CODE",
                'check.equal("Exercise 1", score, 1.25)',
                "score = w * x + b",
            ),
            *_exercise(
                2,
                "def neuron(x, w, b):\n    return ...  # YOUR CODE",
                'check.returns("Exercise 2", neuron, 2.0, 0.5, 0.25, expected=1.25)',
                "def neuron(x, w, b):\n    return w * x + b",
            ),
            ("code", "lesson_score = w * x + b\nprint(lesson_score)"),
        ],
    )
    before = notebook.read_text()

    assert check_exercises.check_notebook(notebook, cell_timeout_s=60) == []
    assert notebook.read_text() == before
    assert check.STRICT_ENV not in __import__("os").environ


def test_exercises_report_a_solution_that_fails_its_check(tmp_path: Path) -> None:
    notebook = _write(
        tmp_path / "01_M" / "01_wrong.ipynb",
        [
            SETUP,
            *_exercise(
                1,
                "score = ...  # YOUR CODE",
                'check.equal("Exercise 1", score, 1.25)',
                "score = w * x",
            ),
        ],
    )

    problems = check_exercises.check_notebook(notebook, cell_timeout_s=60)

    assert len(problems) == 1
    assert problems[0].startswith("[cell 2] with the solutions in place: AssertionError")


def test_exercises_report_a_later_cell_that_uses_the_answer(tmp_path: Path) -> None:
    notebook = _write(
        tmp_path / "01_M" / "01_dependent.ipynb",
        [
            SETUP,
            *_exercise(
                1,
                "score = ...  # YOUR CODE",
                'check.equal("Exercise 1", score, 1.25)',
                "score = w * x + b",
            ),
            ("code", "doubled = 2 * score"),
        ],
    )

    problems = check_exercises.check_notebook(notebook, cell_timeout_s=60)

    assert problems == ["[cell 4] uses `score`, defined in the cell to complete of Exercise 1"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("print(score * 2)", "reads"),
        ("score = 0.5\nprint(score)", "redefines"),
        ("score = score + 1", "reads"),
        ("print(score)\nscore = 0.5", "reads"),
        ("def score():\n    return 1", "redefines"),
        ("def double(score):\n    return 2 * score", None),
        ("def later():\n    return 2 * score", "reads"),
        ("total = sum(score for score in [1, 2])", None),
        ("other = 1", None),
        ("%timeit score", None),
        ("def broken(:", None),
    ],
)
def test_name_use(source: str, expected: str | None) -> None:
    assert check_exercises.name_use(source, "score") == expected


@pytest.mark.parametrize(
    ("cells", "fragment"),
    [
        ([("code", "score = ...  # YOUR CODE")], "must start with `# Exercise N`"),
        (
            [("code", "# Exercise 1\nscore = ...  # YOUR CODE"), ("md", "no check")],
            "must be followed by a code cell that calls",
        ),
        (
            [
                ("code", "# Exercise 1\nscore = ...  # YOUR CODE"),
                ("code", 'check.equal("Exercise 1", score, 1.0)'),
                ("md", "The solution is not collapsed."),
            ],
            "must be followed by a markdown cell",
        ),
        (
            [
                ("code", "# Exercise 1\nscore = ...  # YOUR CODE"),
                ("code", 'check.equal("Exercise 1", score, 1.0)'),
                ("md", SOLUTION.format(block="# Exercise 2\nscore = 1.0")),
            ],
            "must start with `# Exercise 1`",
        ),
        (
            [
                ("code", "# Exercise 1\nscore = ...  # YOUR CODE"),
                ("code", 'check.equal("Exercise 1", score, 1.0)'),
                ("md", SOLUTION.format(block="# Exercise 1\nscore = ...  # YOUR CODE")),
            ],
            "still holds `# YOUR CODE`",
        ),
        ([("md", SOLUTION.format(block="# Exercise 1\nscore = 1.0"))], "no well-formed exercise"),
        ([("code", "x = 1")], "no cell to complete"),
    ],
)
def test_exercises_report_a_problem_of_form(
    tmp_path: Path, cells: list[tuple[str, str]], fragment: str
) -> None:
    notebook = _write(tmp_path / "01_M" / "01_form.ipynb", cells)

    problems = check_exercises.check_notebook(notebook, cell_timeout_s=60)

    assert any(fragment in problem for problem in problems)


def test_exercises_main_exit_codes(tmp_path: Path) -> None:
    _write(tmp_path / "01_M" / "01_none.ipynb", [("code", "x = 1")])

    assert check_exercises.main(["--notebook-dir", str(tmp_path)]) == 1
    assert check_exercises.main(["--notebook-dir", str(tmp_path), "nothing/*.ipynb"]) == 2
