# ArbiterKB × MEMI：RoboMME common-harness 阶段报告

日期：2026-10-30（America/Los_Angeles）

研究分支：`research/perception-belief-robomme`

冻结清单：`robomme-common-harness-v1`，SHA `ec0b1fe2ff7f7027`

## 结论先行

本轮把研究从“离线 perception / belief 对照”推进到了**可执行的同环境
RoboMME closed-loop harness**，并在本机真实运行了四个冻结任务的 reset、语义
观测、官方 `multi_choice` wrapper、运动规划和一个完整高层动作。模拟器不是当前
阻塞项；Apple Silicon 上的 Linux/amd64 CPU fallback 可运行，只是视频任务很慢。

但是，本轮**没有产生 MEMI-vs-ArbiterKB closed-loop 成功率**。原因不是 GPU 或
Vulkan，而是环境中没有明确提供可服务冻结条件 `gpt-6-luna` 的
`OPENAI_API_KEY` / `OPENAI_API_URL`；也没有把此前分享的 OpenRouter 密钥读出、
打印、转存或用于付费调用。因此四臂 API smoke、受控观测退化和 held-out test
均未启动。任何把下面的脚本预检结果写成策略分数的做法都会是错误的。

当前 go/no-go 判定：

- **GO：** common observation、四臂 policy adapter、官方 planner / executor、
  固定 episode reset、记录与统计管线已经具备继续运行的条件。
- **NO-GO：** 在没有刻意提供兼容凭据与 endpoint 前，不启动付费四臂 smoke；
  在 smoke 与 val degradation 通过前，不读取 held-out test。

## 已实现的共同评测条件

四个正式 arm 是：

1. 官方 FullHist；
2. 官方 MEMI（使用锁定 release 的 `optimized_luna.json`）；
3. soft identity repair；
4. full ArbiterKB。

共同条件如下：

- MEMI `bb6c37c`、RoboMME `fae9f73`、ManiSkill `07be6fb`；
- 同一 `gpt-6-luna`、low effort、32,768 completion token 上限；
- 同一 val/test episode、RoboMME seed、reset、可见语义记录、菜单、
  `multi_choice` executor 与最多 20 次决策预算；
- 同一四任务集合：VideoUnmaskSwap、ButtonUnmaskSwap、PickHighlight、
  VideoRepick；锁定版本中 VideoRepick 确实可用；
- val episode 0 做 clean smoke，val 1--9 才做 clean / 15% record dropout /
  4 px centroid jitter，之后才允许 test 0--49；
- 预先声明六个 paired contrasts，不根据 test 结果改比较对象。

FullHist 和 MEMI 直接从官方 checkout 导入。soft 与 ArbiterKB 使用同一个
FullHist 动作 JSON scaffold，但历史由 deterministic memory snapshot 取代。
snapshot 只接收同一批可见 semantic records；同类物体靠 2D/3D 几何连续性形成
匿名 track，不接收 simulator actor identity。目标文字中明确出现而当前被遮住的
彩色 cube 会被预声明，避免“没看见就不存在”的词表缺失。

原始 actor identity 只在决策之后用于 belief truth / Brier 评分，不进入模型提示或
KB 更新。每个正式 row 还会核对四臂的 `reset_digest`、RoboMME seed/difficulty、
源码 SHA、参数 SHA、prompt SHA、commit 与 dirty 状态；不一致即审计失败。

## 运行环境与现实设置路径

原生 macOS 路径被完整检查：SAPIEN 3.0.3 可以导入，但固定 ManiSkill 的运动
规划依赖是 `mplib==0.1.1`，该版本没有 macOS/arm64 wheel；原生 venv 中
`mplib` 不存在。直接换新版 mplib 会改变 benchmark stack，不能作为论文对照。

随后构建了固定 Linux/amd64 CPU 镜像：

- image id：`sha256:07424f3b...59fd1`；
- Python 3.11，Torch 2.9.1 CPU，SAPIEN 3.0.3，mplib 0.1.1；
- Docker daemon 29.7.2；
- Apple Silicon 上用 x86 转译和 CPU renderer。

四个任务都完成 reset，并各报告 50 个 val episodes。首次 reset 诊断耗时分别为
97.29、0.94、0.66、53.69 秒；视频任务的首次加载明显更慢。

## 冻结四任务执行器预检

预检使用无 LLM 的 blind policy，只选择菜单第一项，并且只执行一个高层动作。
其目的仅是验证从观测到 point / choice，再到官方 motion planner、逐帧采集、终止
状态和日志的完整链路。

| 任务 | val seed | 结果 | 墙钟时间 | reset digest |
|---|---:|---|---:|---|
| ButtonUnmaskSwap | 1070000 | preflight_complete | 35.4 s | `edcdb5366a02d0cf` |
| PickHighlight | 1120000 | preflight_complete | 31.4 s | `e5bfc2d7ab8100cb` |
| VideoUnmaskSwap | 1050000 | fail | 91.0 s | `12507d24b522d7e9` |
| VideoRepick | 1090000 | fail | 253.8 s | `00785fa8fc272f65` |

四行均来自 commit `5c89d28`，`arbiter_dirty=false`，模型调用 0，付费 0，
harness error 0。两项 `fail` 是 blind first-menu action 被任务判定为终局失败，
不是模拟器或 harness 崩溃，更不是 FullHist 的成功率。报告器已把该 stage 标记为
`diagnostic_only`，不会计算伪造的 paired policy statistics。

## 统计、成本与泄漏控制

正式 stage 的报告器已经实现：

- 每 arm / task / observation cell 的 Wilson success interval；
- exact paired McNemar 与 Holm correction；
- 成本、模型 latency、调用次数的 10,000 次 paired bootstrap interval；
- belief accuracy、Brier、ECE；
- incomplete arm、跨 arm reset digest、episode seed、dirty tree、harness error 审计。

API 失败不会写成失败 episode；它会停止 stage，避免把基础设施错误当策略错误。
凭据只由官方 client 从环境读取，不写入 result、trace、manifest 或报告。本轮环境
变量检查只记录 present/absent，未显示任何密钥内容。

冻结清单已经于 `2026-10-31T04:38:47Z` 写入 append-only ledger。到本报告为止
没有读取 RoboMME test split，因此没有 test-set tuning。

## 当前精确阻塞与下一闸门

唯一阻止四臂 clean smoke 的条件是：没有刻意提供能服务冻结
`gpt-6-luna` chat-completions 请求的凭据与 endpoint。旧 OpenRouter key 即使存在，
也不能在未验证模型别名、reasoning 参数、stream usage 与计费一致性的情况下被
自动替代进官方条件；那会同时引入花费和不可比性。

凭据条件满足后，下一步不需要再改规则或设计 protocol，直接运行：

1. 四任务 × 四 arm × val episode 0 clean smoke；
2. 审计 16 行的共同 reset、无泄漏、API model id、成本和 latency；
3. 仅 smoke 通过后运行 val 1--9 的三种 observation cells；
4. 保持 manifest 不变，根据预先门槛做一次 go/no-go；
5. 只有 go 才读取冻结的 test 0--49，并如实保留所有负结果。

这意味着下一次工作应当是**运行**，而不是继续扩展 KB 规则。
