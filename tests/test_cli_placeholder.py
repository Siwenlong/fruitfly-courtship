import pytest

from fruitfly_courtship.cli import main


def test_cli_help_lists_analyze_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "analyze" in capsys.readouterr().out
