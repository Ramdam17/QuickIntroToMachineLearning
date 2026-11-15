"""Tests for ``scripts/gen_llms_txt.py`` — the title and purpose read from a notebook header."""

from __future__ import annotations

import json
from pathlib import Path

import gen_llms_txt
import pytest


def write_notebook(folder: Path, header: str, name: str = "01_title.ipynb") -> Path:
    """Write a notebook whose first markdown cell is ``header`` and return its path."""
    cells = [
        {"cell_type": "code", "source": "import numpy as np"},
        {"cell_type": "markdown", "source": header.splitlines(keepends=True)},
        {"cell_type": "markdown", "source": "A later cell that must not be read."},
    ]
    path = folder / name
    path.write_text(json.dumps({"cells": cells}))
    return path


@pytest.mark.parametrize(
    "position_line",
    [
        "*Module 00 · Getting Started — notebook 1 of 11*",
        "*Notebook 1 of 6 — Logistic Regression*",
        "*Notebook 2 of 5 · Support Vector Machines*",
    ],
)
def test_position_line_is_not_the_purpose(tmp_path: Path, position_line: str) -> None:
    header = f"# 01 — The margin\n\n{position_line}\n\nPick the widest street.\n"
    title, purpose = gen_llms_txt.title_and_purpose(write_notebook(tmp_path, header))
    assert title == "01 — The margin"
    assert purpose == "Pick the widest street."


def test_purpose_is_the_whole_first_paragraph_on_one_line(tmp_path: Path) -> None:
    header = "# 02 — Soft margin\n\nSome points sit\non the wrong side.\n\nA second paragraph.\n"
    _, purpose = gen_llms_txt.title_and_purpose(write_notebook(tmp_path, header))
    assert purpose == "Some points sit on the wrong side."


def test_field_labels_lists_and_notes_are_skipped(tmp_path: Path) -> None:
    header = (
        "# 03 — Kernels\n\n"
        "> A quoted note.\n\n"
        "**Prerequisites**\n- module 05, notebook 2\n\n"
        "- a list item\n\n"
        "Bend the space, keep the line.\n"
    )
    _, purpose = gen_llms_txt.title_and_purpose(write_notebook(tmp_path, header))
    assert purpose == "Bend the space, keep the line."


def test_header_without_prose_gives_an_empty_purpose(tmp_path: Path) -> None:
    header = "# 04 — Parameters\n\n*Notebook 4 of 5 · SVM*\n\n**Prerequisites:** none.\n"
    title, purpose = gen_llms_txt.title_and_purpose(write_notebook(tmp_path, header))
    assert title == "04 — Parameters"
    assert purpose == ""


def test_title_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    path = write_notebook(tmp_path, "No heading here.\n", name="05_case.ipynb")
    assert gen_llms_txt.title_and_purpose(path) == ("05_case", "No heading here.")


def test_every_course_notebook_gets_a_purpose_that_names_no_position() -> None:
    lines = [ln for ln in gen_llms_txt.build_llms_txt().splitlines() if ln.startswith("- [")]
    assert lines, "the index lists no notebook"
    for line in lines:
        assert len(line.split(" — ")) >= 3, f"no purpose in: {line[:80]}"
        assert not gen_llms_txt.POSITION_LINE.search(line), f"position line in: {line[:80]}"
