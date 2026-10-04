from pathlib import Path

from verity.cli import main


def test_ask_reports_missing_notes_directory(tmp_path: Path, capsys):
    missing = tmp_path / "missing"

    result = main(["--no-models", "ask", "test question", "--notes", str(missing)])

    output = capsys.readouterr()
    assert result == 2
    assert output.out == ""
    assert f"Error: corpus root is not a directory: {missing}" in output.err
