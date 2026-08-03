"""Serve a trained openpi policy to the inin robot over the inin-stream gRPC protocol.

This is the gRPC counterpart of scripts/serve_policy.py: the robot workstation
keeps using inin-stream's WorkstationStreamClient (schema ``bc-ur5-v2``) and
receives 16x8 absolute_tcp_gripper_chunk actions.

Example:
    uv run scripts/inin_serve.py --config pi05_inin_ur5 \
        --checkpoint /mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_inin_ur5/ur5_v1/29999 \
        --prompt "put the corn in the pan" \
        --server.bind 0.0.0.0:43602
"""

import dataclasses
import logging
import pathlib

from inin_stream.schema import builtin_schema_path
from inin_stream.server import RobotStreamServer
import tyro

from openpi.inin import config as _inin_config
from openpi.inin.serve import InferenceCallbacks
from openpi.inin.serve import warmup as _warmup
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _train_config


@dataclasses.dataclass(frozen=True)
class Args:
    # Training config name used to create the policy (e.g. pi05_inin_ur5).
    config: str = "pi05_inin_ur5"
    # Checkpoint directory (e.g. checkpoints/pi05_inin_ur5/ur5_v1/29999).
    checkpoint: pathlib.Path = tyro.MISSING
    server: _inin_config.StreamServerConfig = dataclasses.field(default_factory=_inin_config.StreamServerConfig)
    # Maximum action-chunk send rate.
    rate_hz: float = 5.0
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

    if args.warmup_iters > 0:
        _warmup(policy, prompt=args.prompt, iterations=args.warmup_iters)

    schema_path = builtin_schema_path(args.server.schema_id)
    callbacks = InferenceCallbacks(policy, prompt=args.prompt, rate_hz=args.rate_hz)
    with _inin_config.make_server_config_dir() as tmp:
        server_yaml = _inin_config.write_server_yaml(tmp, args.server)
        server = RobotStreamServer(server_yaml, schema_path, callbacks)
        callbacks.attach_stream_server(server)
        callbacks.start()
        server.start()
        logging.info("inin inference server ready on %s (rate_hz=%.3f)", args.server.bind, args.rate_hz)
        try:
            server.wait()
        except KeyboardInterrupt:
            logging.info("stopping inin inference server")
            server.stop()
        finally:
            callbacks.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
