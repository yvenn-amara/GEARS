"""The README cannot lie: every code block it marks is executed here.

A block is *marked* by the line ``<!-- test -->`` right above its fence. The marked blocks run in the
order of the README, in one shared namespace (a later block may use what an earlier one defined,
as a reader following the README does), in an empty temporary folder that holds ``data/sessions.csv``
(synthetic sessions). ``python`` blocks are executed; ``bash`` blocks are lists of ``gears ...``
commands, run through the command-line entry point. The official French model is served from a local
file (``french_release``): no download.

Three more rules keep it honest: every ``python`` block must be marked (an unmarked block would be
untested), every file the README links to or names must exist, and it stays under 250 lines.

It is part of the default run: ``pytest tests/test_readme.py``.
"""

from __future__ import annotations

import re
import shlex
import warnings
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from gears.cli import main
from gears.data.loader import make_demo_data

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
MARKED = re.compile(r"<!-- test -->[ \t]*\n```(\w+)[ \t]*\n(.*?)\n```", re.DOTALL)


def marked_blocks() -> list[tuple[int, str, str]]:
    """``(line number, language, code)`` of each marked block, in README order."""
    text = README.read_text(encoding="utf-8")
    return [
        (text.count("\n", 0, m.start()) + 2, m.group(1), m.group(2)) for m in MARKED.finditer(text)
    ]


def commands(script: str) -> list[list[str]]:
    """The commands of a shell block: comments dropped, ``\\`` continuations joined."""
    joined = re.sub(r"\\\n", " ", script)
    lines = [line.strip() for line in joined.splitlines()]
    return [shlex.split(line) for line in lines if line and not line.startswith("#")]


def test_the_readme_has_marked_blocks_in_both_languages():
    languages = {language for _, language, _ in marked_blocks()}
    assert languages == {"python", "bash"}, "the README lost its python or its command-line examples"


def test_every_python_block_of_the_readme_is_marked_and_therefore_tested():
    text = README.read_text(encoding="utf-8")
    fences = len(re.findall(r"^```python[ \t]*$", text, flags=re.MULTILINE))
    marked = sum(1 for _, language, _ in marked_blocks() if language == "python")
    assert fences == marked, "a ```python block has no <!-- test --> line above it"


def test_the_readme_is_short_and_points_only_to_files_that_exist():
    text = README.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 250

    targets = re.findall(r"\]\(([^)#\s]+)\)", text)                 # [label](path)
    targets += re.findall(r"`((?:examples|data|tests|gears)/[\w./-]+)`", text)  # `examples/01_x.py`
    targets += re.findall(r'(?:src|srcset)="([^"]+)"', text)                    # the logo: <img src="assets/...">
    local =[t for t in targets if not t.startswith(("http://", "https://"))]
    assert local, "no local link found: the pattern is broken"
    missing = [t for t in local if not (ROOT / t).exists()]
    assert not missing, f"the README names files that do not exist: {missing}"


@pytest.mark.usefixtures("french_release")
def test_every_marked_block_of_the_readme_runs(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    make_demo_data(n=3000, seed=0).to_csv(tmp_path / "data" / "sessions.csv", index=False)
    monkeypatch.chdir(tmp_path)

    namespace: dict = {"__name__": "readme"}
    runner = CliRunner()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for line, language, code in marked_blocks():
            where = f"README.md, block starting at line {line}"
            if language == "python":
                try:
                    exec(compile(code, f"README.md:{line}", "exec"), namespace)  # noqa: S102 (that is the test)
                except Exception as error:  # noqa: BLE001 (any failure names the block that raised it)
                    pytest.fail(f"{where} failed: {type(error).__name__}: {error}")
            else:
                for argv in commands(code):
                    assert argv[0] == "gears", f"{where}: only `gears` commands are run, not {argv[0]!r}"
                    result = runner.invoke(main, argv[1:])
                    assert result.exit_code == 0, f"{where}: `{' '.join(argv)}` failed:\n{result.output}"

    # what the commands wrote, and what the blocks computed, is there
    assert (tmp_path / "my_model.joblib").exists() and (tmp_path / "simulated.csv").exists()
    assert len(pd.read_csv(tmp_path / "paris.csv")) == 7 * 40          # `gears simulate --pretrained`
    assert len(namespace["simulated"]) > 0 and len(namespace["paris_week"]) == 7 * 40
    assert {"total_energy_kwh", "scenario"} <= set(namespace["daily"].columns)

    # a README that teaches users to ignore warnings is not clean: none of its code may warn
    warned = [str(w.message)[:100] for w in caught if issubclass(w.category, UserWarning)]
    assert warned == []
