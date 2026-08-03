"""Serve dataset-driven open-loop predictions through the synchronous inin protocol.

The real workstation observation is used only as a request and pose-safety
check.  Policy inputs come from one LeRobot episode and advance by
``exec_steps`` after every emitted action chunk.

Dry-run is the default.  Pass ``--execute`` only after the robot has been
aligned with the selected dataset frame and the dry-run logs look correct.
"""

import dataclasses
import logging
import pathlib

from inin_stream.schema import builtin_schema_path
from inin_stream.server import RobotStreamServer
import tyro

from openpi.inin import config as _inin_config
from openpi.inin.serve import warmup as _warmup
from openpi.inin.serve_openloop import LeRobotEpisodeSource
from openpi.inin.serve_openloop import OpenLoopInferenceCallbacks
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _train_config


def _openloop_server_config() -> _inin_config.StreamServerConfig:
    return _inin_config.StreamServerConfig(bind="0.0.0.0:43604")


@dataclasses.dataclass(frozen=True)
class Args:
    config: str = "pi05_ur5_stack_blocks"
    checkpoint: pathlib.Path = tyro.MISSING
    data_root: pathlib.Path = _inin_config.DEFAULT_DATA_ROOT
    repo_id: str | None = None
    episode: int = 0
    start_frame: int = 0
    end_frame: int | None = None
    exec_steps: int = 10
    # Real actions are disabled unless this flag is explicitly supplied.
    execute: bool = False
    max_translation_m: float = 0.03
    max_rotation_deg: float = 15.0
    warmup_iters: int = 2
    server: _inin_config.StreamServerConfig = dataclasses.field(default_factory=_openloop_server_config)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    if args.exec_steps <= 0:
        raise ValueError("exec_steps must be positive")

    train_config = _train_config.get_config(args.config)
    source = LeRobotEpisodeSource.from_train_config(
        train_config,
        episode=args.episode,
        data_root=args.data_root,
        repo_id=args.repo_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.exec_steps,
    )
    first_frame = source.current()
    logging.info(
        "loading open-loop policy config=%s checkpoint=%s prompt=%r",
        args.config,
        args.checkpoint,
        first_frame.prompt,
    )
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        default_prompt=first_frame.prompt,
    )
    if args.warmup_iters > 0:
        _warmup(policy, prompt=first_frame.prompt, iterations=args.warmup_iters)

    callbacks = OpenLoopInferenceCallbacks(
        policy,
        source,
        exec_steps=args.exec_steps,
        execute=args.execute,
        max_translation_m=args.max_translation_m,
        max_rotation_deg=args.max_rotation_deg,
    )
    schema_path = builtin_schema_path(args.server.schema_id)
    with _inin_config.make_server_config_dir() as tmp:
        server_yaml = _inin_config.write_server_yaml(tmp, args.server)
        server = RobotStreamServer(server_yaml, schema_path, callbacks)
        callbacks.attach_stream_server(server)
        callbacks.start()
        server.start()
        logging.warning(
            "inin dataset open-loop server ready on %s mode=%s episode=%d range=[%d,%d) step=%d",
            args.server.bind,
            "EXECUTE" if args.execute else "DRY-RUN",
            args.episode,
            source.start_frame,
            source.end_frame,
            source.frame_step,
        )
        try:
            server.wait()
        except KeyboardInterrupt:
            logging.info("stopping dataset open-loop inference server")
            server.stop()
        finally:
            callbacks.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
