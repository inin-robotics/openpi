# 同步推理模式：client 端适配任务

server 端已经支持一种新的推理模式：**一帧观测换一个动作 chunk，机器人执行完整个 chunk 之后才上传下一帧观测**。
目的是让每个被预测的 chunk 都得到完整执行，不再出现「chunk 还没执行完就被新 chunk 覆盖」的情况。

需要 client 端配合改的只有应用层主循环，本文说明改什么。

## 前提：不需要升级 inin-stream

- 继续用 `inin-stream` 的 `main` 分支，**不需要更新**。
- 协议（`proto/inin_stream.proto`）、`bc-ur5-v2` schema 和 schema hash 全部没有变化，handshake 行为不变。
- 没有新增消息类型。同步节奏完全建立在已有语义上：`ObservationPacket` 当作请求，`ActionMessage` 当作响应。

## 配置改动（一处）

workstation YAML 里把观测队列策略改成 `lossless_blocking`：

```yaml
stream:
  observation_queue_policy: lossless_blocking   # 原来是 latest_only
```

原因：`latest_only` 只保留一个 pending observation，新的会覆盖旧的。同步模式下每一帧观测都是一个必须被应答的请求，一旦被覆盖丢弃，server 永远不会回这个 chunk，client 只能白等一个超时。

`client.run_mode` 建议改成 `bc_infer_sync`。这个字段 server 不做校验，只是让日志能看出对端跑的是哪种模式。

## 主循环契约

```
清空 action buffer
  -> 上传一帧观测
  -> 等待 source_seq_used 等于这一帧 obs_seq 的 chunk
  -> 把整个 chunk 执行完、确认到位
  -> 下一轮
```

关键点是**必须等 chunk 真正执行完毕再上传下一帧观测**。server 端不做限速也不丢帧，它收到几帧观测就回几个 chunk；如果 client 提前上传，就会出现多个 chunk 同时在手上，同步语义失效（server 端会打 `observation backlog` 警告）。

## 等待 chunk 的实现

`ActionMessage.source_seq_used` 会带回 server 推理时用的那一帧的 `obs_seq`，它就是配对用的关联 id。用现有 API 实现：

```python
def wait_for_chunk(stream, obs_seq: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for timed in stream.get_actions_until(time.monotonic_ns()):
            if timed.source_seq_used == obs_seq:
                return timed
            # 不匹配的是上一轮的残留，丢弃
        time.sleep(0.002)
    return None
```

两个容易踩的点：

- `get_actions_until()` 会一次性返回并清空 buffer 里的全部 action（它其实忽略传入的时间参数），所以每次调用都要把返回的整批扫一遍，不能只看第一个。
- 每轮开始前调一次 `stream.clear_action_buffer()`。否则上一轮超时残留的 chunk 会混进来。

主循环大致是：

```python
stream.clear_action_buffer()
frame = stream.prepare_observation(images=..., tensors=..., stamp_ns=..., anchor_observation_key=...)
stream.publish_observation(frame)

timed = wait_for_chunk(stream, frame.obs_seq, timeout_s=ACTION_TIMEOUT_S)
if timed is None:
    ...  # 见下面「超时处理」
execute_chunk_and_wait(timed.action)   # 需要你实现
```

注意 `prepare_observation()` 返回的 `frame` 上就带着这一轮的 `frame.obs_seq`，直接拿来匹配即可。

## chunk 的格式与语义

`timed.action` 是 `(N, 8)` 的 float32 数组：

| 列 | 含义 |
| --- | --- |
| 0:3 | `x, y, z`，TCP 位置 |
| 3:7 | `qx, qy, qz, qw`，TCP 姿态四元数（xyzw 顺序） |
| 7 | `gripper_openness`，0-1 |

- **全部是绝对量**，不是增量。绝对位姿的基准是你上传的那一帧里的 `robot.tcp_pose`。
- `N` 由 server 启动参数 `--exec-steps` 决定，默认 **10**。不要把 N 写死，按 `chunk.shape[0]` 处理。
- 相邻两步的时间间隔是 `1/30` 秒（schema 里 `action_inference.target_period_hz: 30`），第一步对应观测之后的第一个控制周期（`first_target_offset: 1`）。
- `action_mode` 是 `absolute_tcp_gripper_chunk`，和现有异步模式一致。

## 需要 client 端自己实现的部分

这几处依赖机器人控制器，server 端无法代劳：

1. **下发**：把 N 步绝对 TCP 位姿 + gripper 开合送给控制器。
2. **判定执行完毕**：这是同步模式的核心。可以是控制器的运动完成回调、位置误差进入阈值、或者 ROS action 的 result。**不建议只用固定 sleep 兜过去**，那样和异步模式就没区别了。
3. **超时处理**：`wait_for_chunk` 返回 `None` 时怎么办。server 端推理失败时不会发任何通知（这是刻意的设计，避免引入新协议），所以超时是唯一的失败信号。两种合理处置：重新上传当前观测重试一次，或者直接 `stream.abort_episode()` 结束本 episode。建议超时设宽一些（比如 30 秒），因为首帧可能赶上模型 warmup。

## 预期行为

每轮之间会有几百毫秒的停顿：图像上行 + 一次前向（约 100-300 ms）+ chunk 回传。所以机器人的运动是「走一段—停—走一段—停」，这是同步模式的固有代价，不是 bug。

如果觉得停顿占比太高，可以让 server 端把 `--exec-steps` 调大（每轮执行更久，停顿占比下降，但闭环频率变低）；反之调小。这个参数只在 server 端改，client 不用动，因为 client 是「收到多少步就执行多少步」。

## 参考实现与联调

`scripts/inin_fake_workstation.py` 的 `--mode infer_sync` 是一个完整可跑的同步 client，它只用了 `inin-stream` 已发布的 workstation API，可以直接对照。其中 `_wait_for_chunk()` 和 `_run_episode_sync()` 就是上面两段代码的完整版本。

server 端启动方式（默认端口 43603，采集 43601 / 异步推理 43602 之外）：

```bash
uv run scripts/inin_serve_sync.py \
    --config pi05_inin_ur5 \
    --checkpoint <ckpt 目录> \
    --prompt "put the corn in the pan" \
    --server.bind 0.0.0.0:43603 \
    --exec-steps 10
```

prompt 完全由 server 端的 `--prompt` 决定。client 仍然可以在 `episode_start` 里发
`task` metadata（采集流程依赖它），但推理时**不会**被用作 prompt；两者不一致时 server
只打一条 `client episode_start task ... ignored` 警告。所以 client 侧不需要为了推理去
维护指令文本。

联调时可以先看 server 日志：每轮应该有一条 `chunk N for obs_seq=M shape=(10, 8) inference=XXX ms`，`obs_seq` 严格递增且不跳号。如果出现 `observation backlog` 警告，说明 client 没有等 chunk 执行完就上传了下一帧。
