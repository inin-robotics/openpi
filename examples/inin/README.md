# inin UR5 平台接入（采集 / 训练 / 推理）

本目录说明 openpi 与 inin 机器人工作站的对接方式。机器人端继续使用
[inin-stream](../../../inin-stream) 的 `WorkstationStreamClient`（gRPC 双向流 +
schema 哈希握手，schema 为 `bc-ur5-v2`），机器人端代码无需任何修改。

## 架构

```
机器人工作站 (inin-stream WorkstationStreamClient, bc-ur5-v2)
        │ gRPC 双向流（观测上行 / 动作与控制下行）
        ▼
openpi 服务端
  ├─ scripts/inin_collect.py     → LeRobot v2.1 数据集（采集）
  ├─ scripts/compute_norm_stats.py + scripts/train_pytorch.py（训练）
  ├─ scripts/inin_serve.py       → absolute_tcp_gripper_chunk（异步推理，定频）
  ├─ scripts/inin_serve_sync.py  → absolute_tcp_gripper_chunk（同步推理，一帧一 chunk）
  ├─ scripts/inin_serve_openloop.py → 数据集观测驱动的同步 open-loop 推理
  └─ scripts/inin_replay_sync.py → 数据集动作同步回放（无模型，一 stamp 一 chunk）
```

代码位置：

- `src/openpi/inin/collect.py` — 采集回调（有界队列 + 后台写线程、episode 去重、
  commit 回执 `collection_episode_committed`）
- `src/openpi/inin/serve.py` — 异步推理回调（latest-slot + 独立推理线程，不阻塞 gRPC）
- `src/openpi/inin/serve_sync.py` — 同步推理回调（不丢帧队列，一帧观测恰好回一个 chunk）
- `src/openpi/inin/replay_sync.py` — 同步回放回调（不加载模型，一个 ReplayStamp 恰好回一个数据集 chunk）
- `src/openpi/inin/conversions.py` — 表示转换（见下）
- `src/openpi/policies/inin_policy.py` — `IninInputs` / `IninOutputs` 训练变换
- `training/config.py` 中的 `pi05_inin_ur5` 配置

## 表示约定

| 位置 | state | action |
| --- | --- | --- |
| 磁盘 / 网络（bc-ur5-v2 原样） | 14D：`joint(6) + tcp_xyz(3) + tcp_quat_xyzw(4) + gripper(1)` | 8D 绝对：`tcp_xyz(3) + quat_xyzw(4) + gripper(1)` |
| 模型内部 | 7D：`tcp_xyz(3) + rotvec(3) + gripper(1)` | 7D 绝对，经 `DeltaActions` 转 delta（gripper 保持绝对） |

四元数不能直接做逐元素差分，训练时 `IninInputs` 把 quaternion 转为旋转向量，
并以当前 state 的 rotvec 为锚点做连续化（避免跨 ±π 时 2π 跳变污染 delta 和
norm stats）；推理时 `serve.py` 把模型输出的 rotvec chunk 转回 quaternion 下发。

## 端口约定

集群仅开放 **43601–43700**。约定：采集 `43601`，异步推理 `43602`，同步推理 `43603`，
open-loop `43604`，同步回放 `43605`。

## 数据采集

数据根目录：`/mnt/cpfs/zbl-cpfs-new/dataset/harryjhou`（即 `HF_LEROBOT_HOME`），
数据集落在 `<数据根>/<repo_id>`。

```bash
uv run scripts/inin_collect.py \
    --repo-id inin/ur5_bc \
    --task "pick up the corn" \
    --data-root /mnt/cpfs/zbl-cpfs-new/dataset/harryjhou \
    --server.bind 0.0.0.0:43601
```

要点：

- 任务 instruction（训练 prompt）通过 `--task` 参数指定，写入本次采集的每个
  episode，并覆盖工作站 `episode_start` metadata 里的 `task`；不传 `--task`
  时则要求工作站 metadata 自带非空 `task`。
- `episode_end` 必须带 `frames`（帧数），全部帧齐后才 commit 并回发
  `collection_episode_committed`。
- 支持多 task 写入同一数据集。
- 断线续写：进程重启后指向同一 `--repo-id` 即可追加 episode
  （LeRobot v2.1 无 resume API，写入端会重新打开数据集续写）。
- 部分采集的 episode 在退出时会被丢弃，不会写入半截数据。

## 训练（PyTorch）

基座权重（已转换为 PyTorch 格式）：
`/mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_base_pytorch/`，已写入
`pi05_inin_ur5` 配置的 `pytorch_weight_path`。

```bash
export HF_LEROBOT_HOME=/mnt/cpfs/zbl-cpfs-new/dataset/harryjhou

# 1. 归一化统计（写入 assets/pi05_inin_ur5/inin/ur5_bc）
uv run scripts/compute_norm_stats.py --config-name pi05_inin_ur5

# 2. 训练
uv run scripts/train_pytorch.py pi05_inin_ur5 --exp-name <实验名>
```

checkpoint 统一输出在 `/mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_inin_ur5/<实验名>/<step>/`
（已写入 `pi05_inin_ur5` 的 `checkpoint_base_dir`），内含 `model.safetensors`、
`optimizer.pt` 和训练时的 norm stats（`assets/`）。

## 推理

两种模式，协议和 schema 完全相同（`bc-ur5-v2`，`absolute_tcp_gripper_chunk`），
区别只在谁掌握节拍。

### 异步（定频，`scripts/inin_serve.py`）

```bash
uv run scripts/inin_serve.py \
    --config pi05_inin_ur5 \
    --checkpoint /mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_inin_ur5/<实验名>/<step> \
    --prompt "put the corn in the pan" \
    --server.bind 0.0.0.0:43602 \
    --rate-hz 5
```

观测放入 latest-slot（旧观测直接被覆盖，不排队），独立线程跑 `policy.infer()`，
按 `--rate-hz` 限速回发 `action_horizon x 8` 的 chunk（`pi05_inin_ur5` 当前
`action_horizon=50`）。机器人节拍与推理解耦，代价是 chunk 通常执行不完就被新的覆盖。

### 同步（一帧一 chunk，`scripts/inin_serve_sync.py`）

```bash
uv run scripts/inin_serve_sync.py \
    --config pi05_inin_ur5 \
    --checkpoint /mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_inin_ur5/<实验名>/<step> \
    --prompt "put the corn in the pan" \
    --server.bind 0.0.0.0:43603 \
    --exec-steps 10
```

每帧观测恰好回一个 chunk，长度截断到 `--exec-steps`（默认 10，按 schema 的
`target_period_hz: 30` 约 0.33 秒）。机器人执行完整个 chunk 后才上传下一帧观测，
所以每个被预测的 chunk 都完整执行。观测队列不丢帧——丢一帧会让 client 白等一个
超时窗口。client 用 `ActionMessage.source_seq_used` 与自己的 `obs_seq` 配对。

代价是每轮之间有一次完整往返的停顿（上行 + 前向 + 回传），机器人走-停-走-停。
`--exec-steps` 越大停顿占比越低但闭环频率越低。client 端的适配要求见
[sync_client_task.md](sync_client_task.md)。

### 数据集 Open-loop（`scripts/inin_serve_openloop.py`）

该模式保留同步 gRPC 协议和真实 workstation 的 chunk 执行逻辑，但模型输入不使用
机器人上传的图像和 state，而是依次读取指定 LeRobot episode。机器人观测仍用于触发
每轮推理、绑定 `source_seq_used`，以及在下发前校验真实 TCP pose 是否与数据集帧对齐。
每执行一个 `exec_steps` 长的预测 chunk，数据集索引也前进 `exec_steps`。

先以默认 dry-run 检查数据、prompt、推理和位姿误差（不会下发 action）：

```bash
uv run scripts/inin_serve_openloop.py \
    --config pi05_ur5_stack_blocks \
    --checkpoint /mnt/cpfs/zbl-cpfs-new/CKPT/harryjhou/pi05_ur5_stack_blocks/<实验名>/<step> \
    --data-root /mnt/cpfs/zbl-cpfs-new/dataset/harryjhou \
    --episode 0 --start-frame 0 --exec-steps 10 \
    --server.bind 0.0.0.0:43604
```

确认机械臂已经放到数据集起始帧附近、日志中的 pose error 在阈值内后，才增加
`--execute` 允许真实动作：

```bash
uv run scripts/inin_serve_openloop.py \
    --config pi05_ur5_stack_blocks \
    --checkpoint <ckpt> \
    --episode 0 --start-frame 0 --end-frame 200 --exec-steps 10 \
    --max-translation-m 0.03 --max-rotation-deg 15 \
    --server.bind 0.0.0.0:43604 \
    --execute
```

到达 `end-frame`/episode 末尾或 pose 超限时，服务端发送 `stop_remote_action` 并停止
产生新动作。该模式是严格的数据集观测 open-loop：机械臂执行后即使偏离数据集轨迹，
下一轮模型仍会看到后续数据集帧，而不是真实反馈，因此分布偏移会持续累积。真实执行
时应使用较短范围，并保留 workstation 侧限速、工作空间、碰撞和急停保护。

### 在线两种模式共有

prompt 由 `--prompt` 唯一指定，必填。服务端不读客户端 `episode_start` 上传的 `task`
metadata：曾经出现过机器人把运行模式名（`hardware_bc_inference_sync`）填进该字段的情况，
而语言条件化的策略被喂错指令后只会静默退化——表现为沿一个方向匀速漂移、夹爪始终不动。
client 仍可继续发 `task` metadata（采集流程要用），服务端只在它与 `--prompt` 不一致时
打一条 `client episode_start task ... ignored` 警告。

`--prompt` 必须与训练集 `meta/tasks.jsonl` 里的文本逐字一致，大小写和标点都会影响
tokenization。例如 `inin/ur5_corn_in_pan` 是 `put the corn in the pan`。

Open-loop 模式不受此参数影响：它的 prompt 来自被回放的那个 LeRobot episode 自身。

监听端口之前会先用假观测空跑 `--warmup-iters` 次（默认 2，设 0 关闭），把 cuDNN
autotune、首次 PaliGemma 前向和 `resize_with_pad` 的 jit 编译挪到启动阶段；同时显存
不够会在机器人连上之前就直接报错退出。

注意 `pytorch_compile_mode='max-autotune'` 下第一次 warmup 要跑 **1-4 分钟**（inductor
在做 kernel autotune），加上权重加载，从启动到开始监听通常要 5-8 分钟。第二次 warmup
起才接近稳态。所以要先等日志里出现 `server listening` 再让机器人连。

warmup 日志里的单帧耗时可以当作延迟基线：异步模式对照 `--rate-hz`，同步模式下它
直接就是每轮的停顿时长（实测稳态单帧约 80-250 ms）。

## 同步回放（`scripts/inin_replay_sync.py`）

该模式不加载任何模型：机器人每执行完一段就发一个 `ReplayStamp` 要下一段，服务端从指定
LeRobot episode 的 `action` 列里切出下一段绝对位姿回过去。它是同步推理的姊妹模式，骨架
相同，只是把「收观测 → 前向」换成「收 stamp → 切数据」，配对字段从 `obs_seq` 换成
`stamp_seq`（`ActionMessage.source_seq_used` 必须严格相等）。对应的 client 是
`ininfra-bc-replay-sync`。因为不涉及模型，`--prompt` / `--warmup-iters` 这些参数都不存在。

```bash
uv run scripts/inin_replay_sync.py \
    --repo-id inin/ur5_stack_blocks --episode 0 \
    --data-root /mnt/cpfs/zbl-cpfs-new/dataset/harryjhou \
    --start-frame 0 --end-frame 200 --exec-steps 10 \
    --server.bind 0.0.0.0:43605
```

要点：

- 数据来自 LeRobot v2.1 的 `action` 列（`tcp_xyz + quat_xyzw + gripper`，30 Hz 绝对量），
  就是采集时 `actions.tcp_pose_cmd` 与 `actions.gripper_openness_cmd` 拼出来的那 8 列，
  不做任何坐标变换或重锚。
- 第 k 段取第 `k*exec_steps` 到 `k*exec_steps+exec_steps-1` 帧，段与段不重叠、首尾相接
  （训练用的逐帧滑窗在这里会让机械臂每轮回退）。末段不足就发剩余步数，不 padding。
- 游标是 `stamp_seq` 的纯函数，client 每个 episode 从 0 重新计数，所以一条连接里连放多轮
  不需要服务端维护状态。
- 数据放完后**等下一个 stamp 到达时**才发 `replay_complete`；跟着最后一段一起发会被上一轮
  的等待窗口吃掉。
- 取数失败时不发任何消息（不引入新协议），由 client 的 `action_timeout` 兜底。
- 启动只读 parquet 的一列，不 import lerobot（那会拖进整个 torch 栈、约 50 秒），秒级就绪。

**安全**：下发的是数据集里的绝对位姿，服务端没有实时 TCP 可以校验。起始位姿完全靠 client 侧
`reset.fixed_pose` 人工对齐——启动日志会打印首帧的 commanded pose，务必先和 workstation
`configs/bc/replay_sync.yaml` 里的 reset 目标比对，差得远的话第一段就会产生大幅跳跃。换
episode 时要重新确认。首次真机运行建议用 `--end-frame` 限制在很短的范围内。

## 冒烟测试（无真实机器人）

`scripts/inin_fake_workstation.py` 模拟一个符合 bc-ur5-v2 契约的工作站：

```bash
# 采集冒烟：3 个 episode，各 40 帧，等待 commit 回执
uv run scripts/inin_collect.py --data-root /tmp/inin_smoke_home --server.bind 127.0.0.1:43601
uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43601 --mode collect --episodes 3 --frames 40

# 异步推理冒烟：收到 action chunk 即通过
uv run scripts/inin_serve.py --checkpoint <ckpt> --prompt "put the corn in the pan" --server.bind 127.0.0.1:43602
uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43602 --mode infer --episodes 1

# 同步推理冒烟：每帧观测都必须收到配对的 chunk，否则报错退出
uv run scripts/inin_serve_sync.py --checkpoint <ckpt> --prompt "put the corn in the pan" --server.bind 127.0.0.1:43603
uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43603 --mode infer_sync --episodes 1 --frames 20

# 同步回放冒烟：一直放到服务端发 replay_complete，中途配对错位即报错退出
uv run scripts/inin_replay_sync.py --repo-id inin/ur5_stack_blocks --episode 0 --end-frame 45 --server.bind 127.0.0.1:43605
uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43605 --mode replay_sync --episodes 1
```

`--mode infer_sync` 和 `--mode replay_sync` 只使用 `inin-stream` 已发布的 workstation
API，因此它们同时是真实 client 的参考实现。

## 已知限制

- `inin-stream` 生成的 protobuf 代码要求 `protobuf>=6.33` / `grpcio>=1.81`，
  与 `rlds` 依赖组的 `tensorflow-cpu==2.15.0` 冲突，因此 `inin` 与 `rlds`
  两个依赖组在 `uv` 中声明为互斥（DROID RLDS 训练与本链路不能装在同一环境）。
- 数据集为 LeRobot v2.1 图像模式（PNG 逐帧），不是视频模式。
