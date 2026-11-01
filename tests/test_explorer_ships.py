"""The explorer notebook ships with the package and runs in the installed env."""
from pathlib import Path

from click.testing import CliRunner


REPO = Path(__file__).resolve().parents[1]


def test_explore_is_offered_on_the_command_line():
    from qpocket.cli import cli

    result = CliRunner().invoke(cli, ["explore", "--help"])
    assert result.exit_code == 0
    assert "run directory" in result.output.lower()


