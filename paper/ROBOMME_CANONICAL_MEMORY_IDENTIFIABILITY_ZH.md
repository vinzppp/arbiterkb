# ArbiterKB Canonical Memory：零成本可辨识性与研究裁决

状态：**完成；NO-GO — 当前 full KB / scored closed-loop 扩展**

分析提交：`a04a193`

数据范围：G-train 90 episodes、241 identity/cover queries；既有 factorial 与 v2.1 event logs

新 simulator / LLM / API 费用：`0 / 0 / $0.00`

## 摘要

本轮没有为了得到一张成功率表而运行新的闭环。零成本审计发现：ArbiterKB 与强 deterministic ledger 在
241 个 G-train 查询上有 13 次 top-1 分歧（5.39%），其中 7 次涉及 unknown，4 次双方都能发出合法 pick
但 target 不同。这证明 memory state **可以**改变相同 hard-state policy 的合法动作，Track A 的必要条件
窄通过。

但完整证据不支持进入 scored pilot。两臂 top-1 accuracy 完全相同（各 83.40%），13 次分歧中只有 1 次
落在本审计测得的 simultaneous-motion tracker swap 上；Arbiter-only correct 与 ledger-only correct 各 3 次。
Arbiter 的 probability representation 显著减少 deterministic point mass 的 log-loss 惩罚，但 Brier 改善的
episode-cluster interval 跨 0，而且 RoboMME 没有合法的 re-observe / verify / abstain recovery action。
因此 confidence 改善不能在当前动作空间里被识别为闭环 utility。

结论是：**保留最小 canonical projection、tracker error audit 与 phase-safe adapter；不建设 full controller，
不运行 12-cell pilot，不访问 held-out，不启动 degradation。** Full ArbiterKB 的复杂度目前没有被证明必要。

## 1. 输入、比较与泄漏边界

脚本对 G-train suffix `0–9` fail closed，在打开文件前检查；manifest 保存了全部 90 个输入 SHA-256。
G-test 内容没有被打开或复用。已有 held-out 结论只作为历史背景：完整 KB 此前未显示超过 simpler repair，
本轮没有重新消费 held-out。

`strong_ledger` 与 ArbiterKB 接收相同 relinked perceived frames。canonical projection 只读各 memory 已有的
distribution、当前可见 container pixel 和 evidence metadata。oracle arrays 在 decision 完成后才用于评分。

一个重要勘误是：旧 `claim3_g.py` 的 top hypothesis 排除了 `None`，会把本应显式为 unknown 的状态强制成
container。本轮候选空间包含每个 container 与 unknown/elsewhere，避免 forced-answer accuracy。

## 2. State estimation 结果

### 2.1 绝对指标（in-sample diagnostic）

| Memory | top-1 accuracy | mean Brier | mean log loss | ECE-10 | mean pmax | unknown top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Strong ledger | 0.8340 | 0.3382 | 4.5861 | 0.1660 | 1.0000 | 0 |
| ArbiterKB | 0.8340 | 0.3013 | 2.1899 | 0.1071 | 0.9411 | 7 |
| Pilot minimal ledger（诊断） | 0.8257 | 0.3506 | 4.8154 | 0.1743 | 1.0000 | 15 |

deterministic ledger 的概率是 point mass；它不是 calibrated probabilistic forecast。若 point answer 错误，log
loss 会受到很大惩罚。Arbiter 更低的 log loss 证明 distribution representation 在 proper scoring 上有价值，
但不能证明选对动作或机器人成功。

### 2.2 Paired episode-cluster uncertainty

contrast 为 ArbiterKB − strong ledger；90 episode cluster、10,000 bootstrap draws：

| Metric | Estimate | 95% percentile interval |
|---|---:|---:|
| top-1 accuracy | 0.0000 | [−0.0206, +0.0206] |
| mean Brier | −0.0369 | [−0.0831, +0.0079] |
| mean log loss | −2.3961 | [−3.6038, −1.2758] |

这不是 equivalence test，也不是独立 validation。规则曾使用 G-train 开发；bootstrap 只量化这 90 个 episode
内部的 paired sampling uncertainty，不能消除 tuning bias。

## 3. 分歧与合法动作后果

| Paired diagnostic | Count / 241 |
|---|---:|
| top-1 / cover target 不同 | 13 (5.39%) |
| unknown status 不同 | 7 |
| hard actionability 不同 | 7 |
| threshold-0.8 actionability 不同 | 19 |
| 双方合法且 pick target 不同 | 4 |
| correctness 不同 | 6 |
| Arbiter-only / ledger-only correct | 3 / 3 |

4 次不同合法 target 是 hard-state action identifiability 的直接证据；fixture 也验证了相同 state 无论 provenance
或 arm label 都产生相同 action，而改变合法 probability ranking 可以改变 pick target。

19 次 threshold disagreement 不能被当成 19 个更好动作。低 confidence 时 probe 只能 blocked；RoboMME
`multi_choice` 没有“再观察”“验证”“等待”或其他合法 recovery choice。虚构一个动作会改变 benchmark，而不是
公平比较 memory。因此 calibration-to-success gate 失败。

13 次 top-1 分歧中：1 次发生在 measured tracker swap、8 次明确不是 measured swap、4 次 swap status 不可用。
这说明分歧是真实存在的，但不能把它包装成“主要修复了 tracker identity swaps”。

## 4. G-train perception error audit

以下统计来自现有 G-train stress traces，不是 stock RoboMME clean observation 的发生率：

| Measured quantity | Result |
|---|---:|
| Episodes | 90 |
| simultaneous-motion passes | 150 |
| pass 后 slot 落到另一 oracle container | 61 (40.67% / pass) |
| swap kind | exchange 39, merge 12, lost 10 |
| mean episode identity accuracy | 91.79% |
| missed detection | 2,637 / 31,273 oracle-visible frames = 8.43% |
| false association | 2,450 / 28,636 visible-association frames = 8.56% |
| wrong-identity runs / later recovered | 176 / 65 = 36.93% |
| inferred cube occlusion runs | 306 |
| occlusion duration | median 68, p90 137, max 185 frames |

“swap”是 `episode_passes` 的窄定义；“recovery”是 false-association contiguous run 后重新关联初始 container。
这些数字说明现实 tracker error 足以构造未来假设，却不说明 Arbiter 已经能修复它们。事实上 action disagreement
很少与 measured swap 重合。

## 5. Progress bookkeeping 与 `main` bug

### 5.1 精确代码路径

`main@069beb7` 上：

1. `stress/common_episode.py` 在 reset 中把 `video + first` 全部送入 `policy.observe`，之后才
   `cap.start_execution()`；
2. `arbiter_kb/systems/arbiter_kb.py:30-45` 只创建一个 `Detectors`，`step` 不检查 `frame.is_video`；
3. demo 的 grasp/pick/release 因而进入同一个 `kb.events` / `kb.evidence`；
4. `arbiter_kb/views.py:203-208` 对完整 KB history 调用 `progress(kb)`，把 demo action 显示为执行进度；
5. controller 同时看到 shared normalized execution-only line 和 legacy mixed-phase `Progress`。

### 5.2 确定性与记录证据

新增 regression fixture 先只喂 demo grasp。冻结 main-like KB view 显示 `pick-ups 1`，而
`NormalizedTaskState` 正确显示 execution pick-ups 0。测试确认研究 adapter 不再暴露 legacy `Progress` 或 demo
manipulation transcript。

既有 factorial 记录中，有 28 个 decision 同时包含 normalized 与 legacy view，其中 26 个计数冲突。
VideoRepick 的 legacy pick/release 通常正好比 execution-only 多 1，足以污染 cycle completion。

### 5.3 隔离修复

研究分支的 `phase_safe_arbiter_view`：

- 让 shared `NormalizedTaskState` 成为唯一 task-progress authority；
- 从 Arbiter adapter text 删除 legacy `Progress`、`Goal progress` 与重复 manipulation transcript；
- 保留 object、cover、swap、relation 与其他 world-state evidence；
- 不改变 KB 内部、默认 controller view、`main` 或任何冻结 result。

这只是 correctness/fairness 修复，不是概率 KB 有效性的证据。

## 6. Timing reconciliation

“factorial 约 5m49s”是小数位/单位误读。机器记录显示：

| Run | Cells | episode seconds sum | first-to-last event span | mean episode |
|---|---:|---:|---:|---:|
| Factorial | 24 | 3,523.6s = **58m43.6s** | 3,800.7s = **1h03m20.7s** | 146.8s |
| v2.1 validation | 144 | 22,275.7s = **6h11m15.7s** | 23,989.6s = **6h39m49.6s** | 154.7s |

event span 比为 6.31×，cell 数正好为 6×；每 cell 只慢约 5.4%。大差异主要来自 144 对 24 cells，另有
v2.1 的 5 个 600s timeout、更多 task/arm 与更长 action trajectories。

| Recorded component | Factorial mean/cell | v2.1 mean/cell |
|---|---:|---:|
| process/env initialization proxy | 9.71s | 10.17s |
| reset + demo + initial memory feed | 63.91s | 30.98s |
| LLM phase | 7.73s | 13.34s |
| motion planner + executor | 74.88s | 108.75s |
| post-executor capture/conversion/memory | 0.151s | 0.209s |

reset 不能再拆成 simulator reset 与 initial memory processing；post-executor 也把 capture/conversion 与 memory
update 合在一起。日志没有 GPU utilization 或 planner 内部 timing。因此不能据此声称迁移 GPU 必然有用或无用。
可测的主要瓶颈是 simulator reset/demo 与 privileged motion execution，不是 LLM，也不是已记录的增量 memory update。

## 7. Novelty audit

generic structured memory 本身不是 novelty：

- [MEMI](https://memi-robot.github.io/) 已通过 task-optimized structured memory/policy 研究 RoboMME；
- [Ledger](https://arxiv.org/abs/2609.34554) 已提出外部 object ledger；
- [SimpleARM](https://arxiv.org/abs/2609.36595) 已采用 instruction-derived typed relation/reference/progress/route state；
- [Statler](https://arxiv.org/abs/2306.17840) 已让 LLM 显式维护 world state；
- [Persson et al.](https://arxiv.org/abs/1902.09937) 与
  [probabilistic anchoring](https://arxiv.org/abs/2002.10373) 已覆盖 object anchoring、occlusion、multimodal
  distribution 与 probabilistic relational reasoning；
- [GiG](https://proceedings.mlr.press/v306/li26id.html) 用 graph-in-graph experience memory 支持 long-horizon planning；
- [MeMoREval](https://arxiv.org/abs/2610.12561)、
  [EmbodiedMemory-Bench](https://arxiv.org/abs/2609.28236)、
  [WorldLines](https://arxiv.org/abs/2606.18847) 与
  [SpaMEM](https://arxiv.org/abs/2604.22409) 已系统测 memory modality、dynamic state、partial observability 与 belief revision；
- [WatchAct](https://arxiv.org/abs/2606.26443) 明确拆分 video-to-plan、oracle execution 和 integrated success；
- [MemoAct](https://arxiv.org/abs/2603.18494) 已研究 hierarchical short/long-term visuomotor memory。

仍可能成立的窄 gap 是：**在同一真实 tracker error stream、同一 canonical state contract 与同一合法 action
policy 下，做 memory-state → action → success 的 paired causal mediation，并把 calibration utility 与 top-1
identity utility 分开。** 本轮证明这个 gap 值得定义，但当前 RoboMME action space 不能识别 calibration utility，
当前数据也不支持 full ArbiterKB 的增量价值。

## 8. 研究裁决

- **GO — mechanism research（仅窄义）**：hard identity/unknown state 的确能改变合法 pick；保留最小 probe、
  complete candidate scoring、tracker audit 和 phase-safe adapter。
- **NO-GO — full KB**：没有净 top-1 优势，Brier uncertainty 跨 0，正确性 win 3:3，分歧很少落在 measured
  swaps，现有动作空间不能把 confidence 转成合法 recovery。
- **NO-GO — 本阶段 scored pilot**：G-train in-sample、没有 fresh dev set、没有完整 controller validity、
  calibration Track B 不可辨识。

## 9. 最有效的下一 falsification experiment

唯一优先实验应是一个**新采集、未用于调参的 G-like development paired replay**：冻结 tracker 与 error
定义，预先选 VideoUnmaskSwap-like 二选一 legal-pick task；对相同真实 identity-switch/missed-detection stream
运行 strong ledger 与 ArbiterKB，再用同一 deterministic hard-pick controller 执行或 counterfactually score
container choice。primary outcome 是 episode-level correct legal pick，secondary 才是 Brier/log loss/recovery。

若 Arbiter 在 fresh measured-error episodes 上不能以预注册 margin 同时改善 state 与 legal pick，直接否定
“full probabilistic KB 值得继续”的假设。只有该实验先通过，才值得为真实 verify/re-observe action 扩展环境。

最终工程验证：`287 passed`；quality `145 files, 0 problems — PASS`；self-test `16 PASS, 0 FAIL,
1 SKIPPED`（缺少 real oracle replay，沿用仓库预期 skip）；credential pattern scan 为 clean。

## 10. 十个明确答案

1. **零成本 gate 通过了吗？** Track A hard-state 必要条件窄通过；整体 Track B / scored-pilot gate 未通过。
2. **什么条件下分歧？** 241 个 G-train 查询中 13 个 top-1 不同，7 个涉及 unknown；主要不是 measured swap。
3. **哪些分歧改变合法动作？** 4 个双方合法但 pick target 不同；另 7 个 hard actionability 不同。
4. **是否扎根于真实 G-train error？** 输入和 error rate 是真实 G-train；但仅 1/13 分歧落在 measured pass-swap，
   所以不能声称 swap recovery 已被证实。
5. **shared controller 有效吗？** 最小离线 probe 对 action identifiability 有效；完整 task controller 未建设、未验证，
   不能外推机器人成功。
6. **执行 scored pilot 吗？** 没有。calibration 无合法下游动作、无 fresh untuned dev、且无净 top-1 增益。
7. **完整 ArbiterKB 复杂度合理吗？** 当前不合理；保留组件，但 full architecture 为 NO-GO。
8. **`main` 有 demo/execution bug 吗？** 有，代码路径、fixture 和 26/28 recorded conflicts 都确认。
9. **时间差为何巨大？** factorial 实际是约 58m44s episode sum / 63m21s event span，不是 5m49s；
   144 cells 是 24 cells 的 6 倍，event span 6.31 倍，主要是 cell count 与长 executor/timeout。
10. **下一项最有效实验？** fresh G-like development real-error paired replay + identical deterministic legal-pick
    controller；先 falsify state-to-action advantage，再考虑任何闭环或环境扩展。

