from hoi4_agent.cli import smoke
from hoi4_agent.cli.main import main


def test_run_offline_returns_zero(cfg, capsys):
    rc = smoke.run_offline(cfg)
    assert rc == 0
    assert "completed=True" in capsys.readouterr().out


def test_cli_smoke_offline_entrypoint():
    assert main(["smoke-test", "--offline"]) == 0
