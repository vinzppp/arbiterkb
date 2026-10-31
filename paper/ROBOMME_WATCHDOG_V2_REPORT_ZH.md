# ArbiterKB R2：冻结 watchdog 与 RoboMME closed-loop validation 报告

日期：2026-10-31（America/Los_Angeles）

研究分支：`research/perception-belief-robomme`；`main` 未修改。

## 结论先行

R2 已实现并冻结真正的进程级 watchdog。它不依赖阻塞的模型调用或 RoboMME
`env.step()` 返回：每个 cell 在独立进程组运行，supervisor 根据 fsync 的 phase
events 监控 reset、LLM、executor 和整集墙钟；超限时先 SIGTERM，再按冻结 grace
升级 SIGKILL，并由 supervisor 写出结构化终态。

最终冻结协议是 v2.1，manifest SHA `a839ccc592f8aa55`。最初的 v2 SHA
`06f1636d5e387243` 在零 cell 运行时被 pre-run audit 作废：其 reset 上限 180s
低于 v1 VideoRepick preflight 的 reset proxy 205.5s。v2.1 只把统一 reset 上限改为
360s，并用 hash-checked manifest inheritance 保留完整变更链。

无 LLM 的真实 simulator preflight 已在固定 Linux/amd64 镜像中完成：四任务 4/4
都有 result 与 events trace，reset fingerprint 正确，无 timeout、infrastructure
error、incomplete 或 harness error。host 与冻结镜像内的故意挂起自测也都实际杀死
了 LLM、executor、episode 三类阻塞进程，交替动作循环被记录为 diagnostic。

但是，v2.1 的 16-cell API smoke **没有启动**。当前环境没有
`OPENAI_API_KEY` / `OPENROUTER_API_KEY`，请求又明确禁止使用此前共享的 credential。
runner 已验证会在创建 run directory 或发出付费请求前 fail closed。因此本轮 API
调用数和花费均为 0；没有 v2.1 策略胜率，也没有重用 v1 结果冒充新实验。

当前判定：**GO（watchdog implementation）/ NO-GO（16-cell smoke、扩大 validation、
degradation 和 held-out）**。

## v1 证据与勘误

v1 的 15/16 结果和所有负结果保持不变：VideoRepick 只有 MEMI 成功；
ButtonUnmaskSwap 三个完成臂失败；soft cell 在 16:24 后仍未完成。v1 不能支持
任何方法优胜结论。

R2 审查 pinned MEMI `bb6c37c` 后确认官方 `MAX_DECISIONS=50`。旧中文报告写成
“最多 20 次”是不准确的。v2.1 首次在 manifest 中真实强制 20 次：完成 v1 API
episode 最多只有 4 次动作，而 soft 诊断已振荡超过 20 次，因此该阈值基于共同
执行行为，不针对某个 arm 调优。

## 冻结 v2.1 终止协议

| 边界 | 冻结值 | v1 依据 |
|---|---:|---|
| episode wall-clock | 600s | 完成 API episode 最大 198.8s；约 3 倍 |
| reset wall-clock | 360s | VideoRepick preflight reset proxy 205.5s；约 1.75 倍 |
| executor command | 180s | 完成 decision 最大 58.7s；约 3 倍 |
| LLM call 总墙钟 | 120s | 完成 LLM call 最大 12.332s；约 9.7 倍 |
| 单次网络 timeout | 45s | supervisor 另行强制 120s 总墙钟 |
| LLM attempts | 2 | 初次 + 1 次瞬态错误 retry；无 model fallback |
| 高层动作预算 | 20 | 完成 v1 最大 4；soft 诊断超过 20 |

相同规则应用于 FullHist、MEMI、soft repair、ArbiterKB。六次相同动作或三轮 AB
交替只产生 `loop_detected` event，不提前终止，避免 arm-specific termination。

终态和 primary 分析规则：

- `success`、`task_failure`、`timeout` 进入 intention-to-treat；timeout 算失败；
- reset/LLM/API/worker/host/Docker 错误是 `infrastructure_error`，不伪装成任务失败；
- supervisor 被打断后留下 events 但没有 result 时，恢复过程写 `incomplete`，不
  选择性重跑该格；
- 任一 infrastructure error、incomplete、缺格、wrong model、reset mismatch 或
  dirty provenance 都使 smoke NO-GO；
- 每个 run 必须使用新的显式 run ID；同一 run 不重跑已有终态；方法变化需要新
  manifest hash。

冻结 transport 仍是 `openai/gpt-6-luna` 和明确的 OpenRouter chat-completions URL；
endpoint 不匹配或 credential 缺失都会在请求前终止。pinned MEMI 原有 31 attempts
被 v2.1 显式覆盖为最多 2 attempts。

## 真实 simulator preflight

运行 ID：`preflight_20261031T1015Z`。镜像：
`sha256:07424f3b53e08c0bfd27db3a97d89210de92679b040aed3a5aa49e580e059fd1`。

| 任务 | supervisor 总时长 | reset | executor | diagnostic outcome |
|---|---:|---:|---:|---|
| VideoUnmaskSwap | 78.60s | 40.13s | 28.37s | fail |
| ButtonUnmaskSwap | 40.14s | 0.48s | 30.70s | preflight_complete |
| PickHighlight | 42.16s | 0.51s | 31.70s | preflight_complete |
| VideoRepick | 133.59s | 93.95s | 30.74s | fail |

每格 10 个 durable events，包括 episode start/end、reset start/end、policy phase、
executor command/start/end 和 memory transition。四个 reset digest 与 v1 一致；
`arbiter_dirty=false`；manifest SHA 都是 `a839ccc592f8aa55`。两个 `fail` 是 blind
scripted first action 的环境结果，只是 executor 诊断，不是 FullHist 成功率。

## Watchdog 测试与审计

- 故意挂起 LLM phase：进程组被 SIGTERM，终态 `infrastructure_error`；
- 故意挂起 executor phase：进程组被 SIGTERM，终态 `timeout`，primary failure；
- 无 phase event 的整集挂起：episode watchdog 终止并写 `timeout`；
- ABABAB：产生 `alternating_actions`，不改变终止；
- manifest inheritance 会验证父 manifest SHA，错误 SHA 被拒绝；
- v1 reporter 兼容性复核仍只报告原有 Button/soft 缺格，不改变旧结果。

冻结镜像没有 pytest，因此提供了 dependency-free
`scripts/watchdog_selftest.py`；它在 host 和镜像中均 PASS。完整仓库测试与 quality
检查在最终提交前复跑。

## 16-cell smoke 的精确 blocker

环境检查结果：`OPENAI_API_KEY=absent`、`OPENROUTER_API_KEY=absent`。未读取、恢复
或使用此前聊天中的 key。无 credential smoke probe 返回：
`OPENAI_API_KEY is absent; refusing paid/API stage`，且没有创建 run directory。

因此以下 deliverable 本轮不可诚实地产生：16 个 v2.1 模型结果、模型实际返回 ID、
API token/cost、Button/soft 新轨迹，以及 VideoRepick MEMI 优势是否复现。任何用
v1 行回填或凭据缺失时伪造这些内容的做法都会违反协议。

## 下一闸门

只有在运行环境以环境变量提供新的、明确授权的 credential，并把
`OPENAI_API_URL` 精确设为冻结 endpoint 后，才可创建新 run ID，顺序运行完整
16 格。第一处 infrastructure error 会停止 stage；不能只重跑不利格。

若 16/16 都有有效终态、模型 ID/seed/reset/action/provenance 审计通过，report gate
才会授权 `validation`。预注册 validation 是 val episode 1--9、四任务四臂，共
144 cells；预计串行 4.6h、计分 API 成本约 $0.08、停止预算 $0.50。只有该 clean
validation 完成后才重新评估 observation degradation；held-out 仍由独立 gate
保护。

最强可支持结论是：**RoboMME common harness 的不可终止缺口已被可执行、测试过的
统一 watchdog 修复，且 simulator 集成已通过；但由于没有授权 API credential，
尚无新的 closed-loop 方法比较证据，ArbiterKB 的复杂性价值仍未建立。**
