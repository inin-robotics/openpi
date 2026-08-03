"""Serve a trained openpi policy to the inin robot in synchronous chunk mode.

Unlike scripts/inin_serve.py, which sends chunks at a fixed rate from the
latest observation, this entry point answers every observation with exactly one
chunk. The robot publishes an observation, executes the whole chunk it gets
back, and only then publishes again, so every predicted chunk runs to
completion. See examples/inin/sync_client_task.md for the client contract.

Example:
    uv run scripts/inin_serve_sync.py --config pi05_inin_ur5 \
        --checkpoint /mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_inin_ur5/ur5_v1/29999 \
        --prompt "put the corn in the pan" \
        --server.bind 0.0.0.0:43603 --exec-steps 10
"""

import dataclasses
import logging
import pathlib

from inin_stream.schema import builtin_schema_path
from inin_stream.server import RobotStreamServer
import tyro

from openpi.inin import config as _inin_config
from openpi.inin.serve import warmup as _warmup
from openpi.inin.serve_sync import SyncInferenceCallbacks
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _train_config


def _sync_server_config() -> _inin_config.StreamServerConfig:
    return _inin_config.StreamServerConfig(bind="0.0.0.0:43603")


@dataclasses.dataclass(frozen=True)
class Args:
    # Training config name used to create the policy (e.g. pi05_inin_ur5).
    config: str = "pi05_inin_ur5"
    # Checkpoint directory (e.g. checkpoints/pi05_inin_ur5/ur5_v1/29999).
    checkpoint: pathlib.Path = tyro.MISSING
    server: _inin_config.StreamServerConfig = dataclasses.field(default_factory=_sync_server_config)
    # Chunk steps sent per observation. The chunk is truncated here, so the
    # robot simply executes everything it receives. At the schema's 30 Hz
    # target period this sets the closed-loop interval (10 steps ~ 0.33 s).
    exec_steps: int = 10
    # The only source of the inference prompt; the robot's episode_start
    # metadata is never used. Must match the training instruction verbatim
    # (see the dataset's meta/tasks.jsonl).
    prompt: str = tyro.MISSING
    # Synthetic inferences run before the server accepts clients; 0 disables warmup.
    warmup_iters: int = 2


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    train_config = _train_config.get_config(args.config)
    logging.info("loading policy config=%s checkpoint=%s", args.config, args.checkpoint)
    logging.info("serving prompt=%r", args.prompt)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint, default_prompt=args.prompt)

    # Warmup matters more here than in the async server: without it the robot
    # pays for cuDNN autotuning and the resize_with_pad trace while it is
    # already stopped and waiting for its first chunk.
    if args.warmup_iters > 0:
        _warmup(policy, prompt=args.prompt, iterations=args.warmup_iters)

    schema_path = builtin_schema_path(args.server.schema_id)
    callbacks = SyncInferenceCallbacks(
        policy,
        prompt=args.prompt,
        exec_steps=args.exec_steps,
    )
    with _inin_config.make_server_config_dir() as tmp:
        server_yaml = _inin_config.write_server_yaml(tmp, args.server)
        server = RobotStreamServer(server_yaml, schema_path, callbacks)
        callbacks.attach_stream_server(server)
        callbacks.start()
        server.start()
        logging.info("inin sync inference server ready on %s (exec_steps=%d)", args.server.bind, args.exec_steps)
        try:
            server.wait()
        except KeyboardInterrupt:
            logging.info("stopping inin sync inference server")
            server.stop()
        finally:
            callbacks.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
