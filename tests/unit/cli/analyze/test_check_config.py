import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from harbor.analyze.checker import run_checks
from harbor.cli.main import app
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig as WrapperConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_check_keeps_config_and_applies_explicit_overrides(tmp_path, override):
    task = tmp_path / "subject"
    task.mkdir()
    (task / "instruction.md").write_text("Write a result.")
    (task / "task.toml").write_text("")
    for name in ("tests", "solution", "environment"):
        (task / name).mkdir()
        (task / name / "private.txt").write_text(name)
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    config = JobConfig(
        agents=[
            AgentConfig(
                import_path="example:Agent",
                model_name="test-model",
                override_timeout_sec=37,
            )
        ],
        environment=EnvironmentConfig(
            import_path="example:Environment", kwargs={"pinned": True}
        ),
    )
    config.retry.max_retries = 0
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "environment": {
                    "docker_image": "review:agent",
                    "workdir": "/app",
                    "cpus": 2,
                },
                "verifier_environment": {
                    "docker_image": "review:verifier",
                    "workdir": "/app",
                },
            }
        )
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    async def create(actual):
        assert actual.retry.max_retries == 0
        assert actual.agents[0].override_timeout_sec == 37
        assert actual.environment.kwargs == {"pinned": True, "extra": 2}
        if override:
            assert actual.environment.import_path is None
            assert actual.environment.type.value == "docker"
            assert actual.agents[0].name == "nop"
            assert actual.agents[0].import_path is None
        else:
            assert actual.environment.import_path == "example:Environment"
            assert actual.agents[0].import_path == "example:Agent"
            assert actual.agents[0].model_name == "test-model"
        wrapper = actual.tasks[0].path
        cfg = WrapperConfig.model_validate_toml((wrapper / "task.toml").read_text())
        assert cfg.environment.docker_image == "review:agent"
        assert cfg.verifier.environment.docker_image == "review:verifier"
        assert cfg.artifacts[0].source == "/app/check-result.json"
        for name in ("tests", "solution", "environment"):
            assert (
                wrapper / "environment/task" / name / "private.txt"
            ).read_text() == name
        assert (wrapper / "tests/validate.py").is_file()
        return SimpleNamespace(
            job_dir=job_dir,
            run=AsyncMock(return_value=SimpleNamespace(trial_results=[])),
        )

    from harbor.models.environment_type import EnvironmentType

    with patch("harbor.job.Job.create", side_effect=create):
        await run_checks(
            task,
            config_path=path,
            environment_config_path=environment,
            environment_kwargs={"extra": 2},
            agent="nop" if override else None,
            environment=EnvironmentType.DOCKER if override else None,
        )


def test_cli_forwards_config_without_overriding_defaults(tmp_path):
    from harbor.cli.quality_checker.models import CheckReport

    with patch(
        "harbor.analyze.checker.run_checks",
        AsyncMock(return_value=(CheckReport(results=[]), tmp_path)),
    ) as run:
        result = CliRunner().invoke(
            app,
            [
                "check",
                str(tmp_path),
                "--environment-config",
                str(tmp_path / "env.json"),
            ],
        )
    assert result.exit_code == 0
    assert run.call_args.kwargs["agent"] is None
    assert run.call_args.kwargs["environment"] is None
    assert run.call_args.kwargs["environment_config_path"] == tmp_path / "env.json"
