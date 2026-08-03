"""Shared configuration for the inin gRPC collection / serving entry points."""

import dataclasses
import os
import pathlib
import tempfile

import yaml

# Collection writes the LeRobot dataset to <data_root>/<repo_id>. openpi's
# data loader resolves datasets from HF_LEROBOT_HOME/<repo_id>, so training
# and norm-stats jobs must run with HF_LEROBOT_HOME=<data_root>.
DEFAULT_DATA_ROOT = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "/mnt/cpfs/zbl-cpfs-new/dataset/harryjhou"))
DEFAULT_SCHEMA_ID = "bc-ur5-v2"
DEFAULT_REPO_ID = "inin/ur5_bc"


@dataclasses.dataclass(frozen=True)
class StreamServerConfig:
    """gRPC transport settings shared by collection and serving."""

    # gRPC bind address for the inin-stream robot connection.
    # Ports must stay within the 43601-43700 range allowed on this cluster.
    # Convention: 43601 collection, 43602 async serving, 43603 sync serving,
    # 43604 open-loop, 43605 replay. Each entry point overrides this default.
    bind: str = "0.0.0.0:43601"
    # Versioned built-in inin-stream schema id (defines the wire contract).
    schema_id: str = DEFAULT_SCHEMA_ID
    max_send_message_mb: int = 64
    max_receive_message_mb: int = 64


def write_server_yaml(directory: str | pathlib.Path, config: StreamServerConfig) -> pathlib.Path:
    """Materialize the RobotStreamServer YAML config expected by inin-stream."""
    path = pathlib.Path(directory) / "server.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "server": {
                    "bind_address": config.bind,
                    "insecure": True,
                    "max_send_message_mb": config.max_send_message_mb,
                    "max_receive_message_mb": config.max_receive_message_mb,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def make_server_config_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="openpi_inin_")
