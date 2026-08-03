"""Replay a recorded LeRobot episode to the inin robot in synchronous chunk mode.

This is scripts/inin_serve_sync.py with the policy taken out: the client asks
for one chunk at a time with a ReplayStamp instead of an observation, and the
answer is the next contiguous slice of a recorded trajectory. No checkpoint is
loaded and no inference runs, so startup is a parquet read.

SAFETY: the chunks are absolute base-frame poses straight from the dataset, and
the server has no live TCP to re-anchor them against. The starting pose is
aligned by hand on the client through its ``reset.fixed_pose``; if the selected
episode starts far from that pose, the first chunk commands a large jump. The
first frame's TCP pose is logged at startup so it can be compared beforehand.

Example:
    uv run scripts/inin_replay_sync.py \
        --repo-id inin/ur5_stack_blocks --episode 0 \
        --server.bind 0.0.0.0:43605 --exec-steps 10
"""

import dataclasses
import logging
import pathlib

from inin_stream.schema import builtin_schema_path
from inin_stream.server import RobotStreamServer
import tyro

from openpi.inin import config as _inin_config
from openpi.inin.replay_sync import ReplaySyncCallbacks
from openpi.inin.replay_sync import load_episode_actions


def _replay_server_config() -> _inin_config.StreamServerConfig:
    return _inin_config.StreamServerConfig(bind="0.0.0.0:43605")


@dataclasses.dataclass(frozen=True)
class Args:
    # LeRobot dataset root; the episode is read from <data_root>/<repo_id>.
    data_root: pathlib.Path = _inin_config.DEFAULT_DATA_ROOT
    repo_id: str = tyro.MISSING
    episode: int = 0
    # Frame range within the episode, useful for keeping a first real run short.
    start_frame: int = 0
    end_frame: int | None = None
    # Frames replayed per stamp. The client executes the whole chunk before
    # asking for the next one, so this sets how coarse an operator's
    # intervention granularity is (10 steps ~ 0.33 s at the schema's 30 Hz).
    exec_steps: int = 10
    server: _inin_config.StreamServerConfig = dataclasses.field(default_factory=_replay_server_config)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    actions = load_episode_actions(args.data_root, args.repo_id, args.episode)
    callbacks = ReplaySyncCallbacks(
        actions,
        exec_steps=args.exec_steps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    logging.info(
        "loaded replay source repo=%s episode=%d frames=%d range=[%d,%d) exec_steps=%d rounds=%d",
        args.repo_id,
        args.episode,
        actions.shape[0],
        callbacks.start_frame,
        callbacks.end_frame,
        args.exec_steps,
        callbacks.total_rounds,
    )
    first = actions[callbacks.start_frame]
    logging.warning(
        "first commanded pose xyz=[%.4f %.4f %.4f] quat_xyzw=[%.4f %.4f %.4f %.4f] gripper=%.3f; "
        "confirm the client's reset.fixed_pose is close to it before replaying",
        *first[:3],
        *first[3:7],
        first[7],
    )

    schema_path = builtin_schema_path(args.server.schema_id)
    with _inin_config.make_server_config_dir() as tmp:
        server_yaml = _inin_config.write_server_yaml(tmp, args.server)
        server = RobotStreamServer(server_yaml, schema_path, callbacks)
        callbacks.attach_stream_server(server)
        callbacks.start()
        server.start()
        logging.info("inin synchronous replay server ready on %s", args.server.bind)
        try:
            server.wait()
        except KeyboardInterrupt:
            logging.info("stopping inin synchronous replay server")
            server.stop()
        finally:
            callbacks.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
