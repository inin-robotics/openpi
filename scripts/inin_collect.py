"""Run the inin gRPC data-collection server writing a LeRobot v2.1 dataset.

The robot workstation connects with inin-stream's WorkstationStreamClient
(schema ``bc-ur5-v2``); committed episodes land in ``<data_root>/<repo_id>``.
Train with ``HF_LEROBOT_HOME=<data_root>`` so openpi's loader finds the data.

Example:
    uv run scripts/inin_collect.py --repo-id inin/ur5_bc \
        --task "pick up the corn" --server.bind 0.0.0.0:43621
"""

import dataclasses
import logging
import pathlib

from inin_stream.schema import builtin_schema_path
from inin_stream.schema import load_schema
from inin_stream.server import RobotStreamServer
import tyro

from openpi.inin import config as _config
from openpi.inin.collect import CollectionCallbacks


@dataclasses.dataclass(frozen=True)
class Args:
    # LeRobot repo id; the dataset is written to <data_root>/<repo_id>.
    repo_id: str = _config.DEFAULT_REPO_ID
    # Dataset root; must match HF_LEROBOT_HOME used at training time.
    data_root: pathlib.Path = _config.DEFAULT_DATA_ROOT
    # Task instruction written to every collected episode (used as the training
    # prompt). Overrides any task in the workstation's episode_start metadata;
    # if omitted, the workstation must provide one.
    task: str | None = None
    server: _config.StreamServerConfig = dataclasses.field(default_factory=_config.StreamServerConfig)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    # Import lerobot (torch, datasets, ...) before accepting connections so the
    # first episode is not delayed by a cold import inside the writer thread.
    logging.info("warming up lerobot import")
    import lerobot.common.datasets.lerobot_dataset  # noqa: F401

    schema_path = builtin_schema_path(args.server.schema_id)
    schema = load_schema(schema_path)
    logging.info(
        "starting inin collection server bind=%s schema=%s dataset=%s",
        args.server.bind,
        args.server.schema_id,
        args.data_root / args.repo_id,
    )
    callbacks = CollectionCallbacks(
        schema=schema,
        data_root=args.data_root,
        repo_id=args.repo_id,
        task=args.task,
        effective_config={
            "repo_id": args.repo_id,
            "data_root": str(args.data_root),
            "task": args.task,
            "server": dataclasses.asdict(args.server),
        },
    )
    with _config.make_server_config_dir() as tmp:
        server_yaml = _config.write_server_yaml(tmp, args.server)
        server = RobotStreamServer(server_yaml, schema_path, callbacks)
        callbacks.attach_stream_server(server)
        server.start()
        logging.info("inin collection server ready; waiting for workstation connection")
        try:
            server.wait()
        except KeyboardInterrupt:
            logging.info("stopping inin collection server")
            server.stop()
        finally:
            callbacks.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
