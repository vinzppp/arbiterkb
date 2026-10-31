# ArbiterKB × MEMI：RoboMME common-harness validation 报告

日期：2026-10-31（America/Los_Angeles）

研究分支：`research/perception-belief-robomme`（未修改 `main`）

冻结清单：`robomme-common-harness-v1-openrouter`，SHA `507cbdd26bf429e7`

## 结论先行

本轮首次真实运行了同一 RoboMME closed-loop harness 下的 FullHist、官方 MEMI、
soft identity repair 和完整 ArbiterKB。四个冻结 val episode 0 任务中，15/16 个
arm-task 格得到可评分结果；所有返回模型均为 `openai/gpt-6-luna`，使用同一官方
`multi_choice` executor、reset、观测、seed 和任务菜单。

当前闸门判定是 **NO-GO 进入 observation degradation，更不读取 held-out test**。
原因不是策略胜率，而是 `ButtonUnmaskSwap/soft` 在冻结计分运行中持续 16 分 24 秒
未返回，且协议只冻结了单次 LLM timeout，没有冻结 episode/executor wall-clock
终止语义。该格没有结果文件，也没有被伪造成失败。独立、不计分的诊断复现运行了
12 分钟、下发 22 个均可返回的 A/B 按钮动作，但策略持续振荡，证明 episode 可以
产生极长尾；诊断结果没有回填计分表。

在 15 个完成格上，MEMI 为 3/4，FullHist 为 2/4，ArbiterKB 为 2/4，soft 为
2/3（另 1 格缺失）。样本只有每任务一个 episode，且 soft 不完整；所有预注册
paired exact McNemar 检验的 Holm 校正后 `p=1.0`。这些数值只能证明 harness 已经
跑通并暴露了真实负结果，不能支持任何 arm 优于另一 arm 的论文结论。

## 冻结 clean smoke 结果

下表成本按冻结清单中的 Luna 单价从完成行 token 估算，仅含已落盘的模型调用。

| 任务 | arm | outcome | 调用 | 墙钟 | 估算成本 |
|---|---|---|---:|---:|---:|
| VideoUnmaskSwap | FullHist | success | 1 | 85.1 s | $0.000196 |
| VideoUnmaskSwap | MEMI | success | 1 | 100.1 s | $0.000443 |
| VideoUnmaskSwap | soft | success | 1 | 98.8 s | $0.000194 |
| VideoUnmaskSwap | ArbiterKB | success | 1 | 94.4 s | $0.000289 |
| ButtonUnmaskSwap | FullHist | fail | 3 | 105.7 s | $0.000408 |
| ButtonUnmaskSwap | MEMI | fail | 3 | 109.9 s | $0.001445 |
| ButtonUnmaskSwap | soft | **unscored / incomplete** | — | >984 s | 未落盘 |
| ButtonUnmaskSwap | ArbiterKB | fail | 3 | 120.0 s | $0.000634 |
| PickHighlight | FullHist | success | 2 | 84.3 s | $0.000227 |
| PickHighlight | MEMI | success | 2 | 77.6 s | $0.000560 |
| PickHighlight | soft | success | 4 | 118.1 s | $0.000501 |
| PickHighlight | ArbiterKB | success | 2 | 68.0 s | $0.000298 |
| VideoRepick | FullHist | fail | 1 | 140.9 s | $0.000269 |
| VideoRepick | MEMI | success | 3 | 198.8 s | $0.001332 |
| VideoRepick | soft | fail | 1 | 135.4 s | $0.000324 |
| VideoRepick | ArbiterKB | fail | 3 | 187.3 s | $0.000573 |

15 个完成格合计 68,829 input tokens、4,453 output tokens、92.019 秒模型延迟、
1,724.4 秒端到端墙钟，计分行估算成本 $0.007694。被中止的计分运行和诊断运行
没有完成行，因而不在该成本估算中，不能声称这是 OpenRouter 实际总扣费。

## 可辩护的观察与负结果

- VideoUnmaskSwap 四臂都成功，无法区分 memory 方案。
- ButtonUnmaskSwap 的三个完成臂全部失败；完整 KB 没有修复该任务。
- PickHighlight 四臂都成功；ArbiterKB 在这一个 episode 上墙钟最短，但单例不能
  形成速度优势结论。
- VideoRepick 只有 MEMI 成功；这是对 ArbiterKB 不利但必须保留的负结果。
- MEMI 的完成行平均模型延迟和成本最高；相对 ArbiterKB，配对均值差为
  +6.518 秒模型延迟、+$0.000496/episode，但只有 4 对，不能做推广性结论。
- 当前结果没有任何 belief score，因为这些完成调用没有产生可评分的
  `scored_beliefs`；不能用 task success 反推 belief quality。

## 公平性、复现与泄漏审计

- 固定源码：MEMI `bb6c37c`、RoboMME `fae9f73`、ManiSkill `07be6fb`；固定镜像
  `sha256:07424f3b53e08c0bfd27db3a97d89210de92679b040aed3a5aa49e580e059fd1`。
- 固定模型与 transport：请求和返回均为 `openai/gpt-6-luna`，OpenRouter
  chat-completions endpoint；密钥只从环境读取，仓库 secret scan 无命中。
- 四任务 seed/difficulty 与 reset digest 在已完成的共同格中一致：
  `12507d24b522d7e9`、`edcdb5366a02d0cf`、`e5bfc2d7ab8100cb`、
  `00785fa8fc272f65`。
- 15 行均为 `arbiter_dirty=false`，参数 SHA `fe4e94112a74`、prompt SHA
  `307eaf7ff479aeae`。首个 FullHist 行来自冻结后 commit `4a888c3`；其余 14 行
  来自仅修复 structured belief hook 的 `e8d5785`。该修复不改变 FullHist policy、
  观测、executor 或评分，且没有重跑/替换首行。
- 官方 FullHist/MEMI policy 原先不暴露实验 arm，导致原始 JSON 的 `arm` 为
  `unknown`。报告器现在只在该字段为空/unknown 时从不可歧义的结果目录后缀恢复，
  并保留 `arm_recorded`；原始结果未改写。
- actor identity 仍只允许在决策后 truth scoring 使用，不进入模型 prompt 或 memory。
  本轮没有新增 RoboMME test split 读取，`results/test_reads.log` 也没有 test READ。

报告审计的唯一 issue 是 `ButtonUnmaskSwap` 缺少 soft 行；无 dirty row、无
`harness_error`。API、Docker 或人工中止均未被计为策略失败。

## 阻塞与下一闸门

冻结 v1 协议无法一致地处理“策略持续给出合法动作但 episode 极长”与“单次
executor 永不返回”。直接把缺失格算失败、重跑到成功、或只给 soft 加 timeout，
都会引入后验偏差。

进入 degradation 前需要新版本协议，并重新跑完整 clean smoke：

1. 在看到任何 held-out 前预注册统一的 episode wall-clock watchdog；
2. 明确定义 `executor_timeout` / `episode_timeout` 是否作为任务失败，以及资源统计
   如何截尾；四臂必须完全相同；
3. 每次 executor 调用前输出非敏感的 choice/point 进度，超时仍能精确定位；
4. 用新 manifest hash 写入 ledger，重跑 16/16 clean 格并要求 audit pass；
5. 只有该 gate 通过，才运行 val 1--9 的 clean/dropout/jitter；之后仍需独立
   go/no-go 才能读取 test 0--49。

因此下一步仍然是修复和冻结评测终止语义，而不是扩展 KB 规则。
