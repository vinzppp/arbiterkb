# G1 报告：「Memory or policy?」pilot

日期：2026-11-02
分支：`research/perception-belief-robomme`；运行代码 `ac55555`（所有 225 行一致，0 行 dirty）；`main` 仍为 `069beb7`
运行：`g1_20261101T2128Z`；manifest `experiments/memory_or_policy_g1_v1.json`，sha `75ee3199ac0853fe`（已冻结、登记）
gate：`results/robomme_common/memory_or_policy_g1/gate.json`，GO，0 个问题
机器可读：`results/robomme_common/memory_or_policy_g1/g1_20261101T2128Z/g1_report.json`（`scripts/g1_report.py`）

**G1 不为任何假设计分。** 它只测 discordance d、检查公平与终止路径，然后定 n。下面的成功率只为说明 d 从哪来。

## 结论先行

1. **d：** P2a（MM-m vs LD-m）0.52，P2b（KB-m vs LD-m）0.32，P2c（FH-m vs LD-m）0.44。最大 d = 0.52
   （25 对，Wilson 95% 区间 0.34–0.70）。
2. **n：** 按预注册，n_10 = 690 对（d = 0.52），超过 test 的供应 250。**建议 n = 250**（5 任务 × test 0–49，
   即 n = min(250, 供应)）。n = 250 时，80% 功效、Holm 最坏情况下能检测的最小差异约 **16.5 pp**（d 取区间两端时
   13.3–19.2 pp）。按预注册，这个数要在 test 读取前写进 manifest；小于它的结果标为 inconclusive。
   n_TOST = 1,780，远超 250，**H2 的等价形式不能主张**（预注册的条件是 d ≤ 0.07）。
3. **终止路径：** 走到了成功、任务失败、50 次调用上限三条；30 min episode 超时、300 s 执行器超时、无效回答、
   基础设施错误这四条**没有走到**（§3）。
4. **公平：** 在真实请求上成立：同一任务同一策略的策略文本在 4 个记忆臂间 sha 相同；同一记忆的记忆说明在两种
   策略间 sha 相同；1,789 次调用全部 valid、全部返回 `openai/gpt-6-luna`；每个 (任务, episode) 的 9 个臂 reset
   digest 相同。
5. **花费与耗时：** $0.352（另 endpoint probe $0.0000053），远低于 $2；episode 时长合计 33,180 s（9.2 h），
   墙钟 9 h 57 min（2026-11-01 21:28 → 11-02 07:25 UTC），在 G0 估计的 13–27 h 之下。

## 1. 每臂成功数（25 对；不计分，仅说明）

| 臂 | RouteStick | SwingXtimes | VideoUnmaskSwap | StopCube | VideoRepick | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| fh_g | 0 | 0 | 2 | 5 | 2 | 9 |
| fh_m | 2 | 3 | 2 | 4 | 2 | 13 |
| ld_g | 0 | 1 | 5 | 4 | 2 | 12 |
| ld_m | 1 | 3 | 5 | 1 | 4 | 14 |
| mm_g | 4 | 2 | 4 | 0 | 2 | 12 |
| mm_m | 4 | 5 | 3 | 4 | 3 | 19 |
| kb_g | 2 | 0 | 5 | 0 | 3 | 10 |
| kb_m | 2 | 2 | 5 | 0 | 1 | 10 |
| memi | 5 | 4 | 5 | 5 | 4 | 23 |

每格 5 个 episode，这些数字的不确定性很大，不做解读。两点需要在正式验证前知道：

- **split fidelity（MM-m − MEMI）：** 19 vs 23，−16 pp（d = 0.24）。预注册说超过 ±10 pp 就意味着替换或框架起作用，
  H3 要带这条说明。G1 的 n 太小，不能下结论，但点估计已越过 10 pp，正式验证时这条注释很可能需要。
- **VideoUnmaskSwap** 的 G1 episode（10 12 13 14 18）都是单目标 goal，比 two-pick 容易（orchestrator 已注明）；
  VUS 上 5 个记忆臂里 3 个满分，P2b 在 VUS 上 0/5 不一致，符合「d 偏低」的预期。

## 2. Discordance（预注册 P2 与参考对比）

| 对比 | 前者独胜 | 后者独胜 | d | 各任务不一致数（/5）RS / SX / VUS / SC / VR |
|---|---:|---:|---:|---|
| **P2a** MM-m vs LD-m | 9 | 4 | **0.52** | 3 / 2 / 2 / 3 / 3 |
| **P2b** KB-m vs LD-m | 2 | 6 | **0.32** | 1 / 3 / 0 / 1 / 3 |
| **P2c** FH-m vs LD-m | 5 | 6 | **0.44** | 1 / 2 / 3 / 3 / 2 |
| split fidelity MM-m vs MEMI | 1 | 5 | 0.24 | 1 / 1 / 2 / 1 / 1 |
| FH-m vs FH-g | 5 | 1 | 0.24 | 2 / 3 / 0 / 1 / 0 |
| MEMI vs FH-g | 14 | 0 | 0.56 | 5 / 4 / 3 / 0 / 2 |

n 的计算按预注册：n_10 = (2.81√d + 0.84√(d − 0.01))² / 0.01，取最大 d。

| d | n_10 | n = 250 时可检测的差异 |
|---|---:|---:|
| 0.335（区间下端） | 443 | 13.3 pp |
| 0.32（P2b） | 423 | 13.0 pp |
| 0.44（P2c） | 583 | 15.2 pp |
| **0.52（最大，P2a）** | **690** | **16.5 pp** |
| 0.70（区间上端） | 929 | 19.2 pp |

d 远高于预注册设想的 0.20：在 25 对上，不同记忆臂的成败常常不一致。正式验证在 250 对上只能检出约 16–17 pp
以上的记忆差异；预注册里 H2 的 5 pp 等价边界和 10 pp 检测目标都达不到。是否接受这个 n，或者缩小家族（例如
只保留 P1 与 P3 作主检验），是 orchestrator / 用户的决定；我不改规则。

## 3. 终止路径

| 路径 | 次数 | 来源 |
|---|---:|---|
| 成功（environment_success） | 122 | — |
| 任务失败（环境判 fail） | 87 | — |
| 50 次调用上限（`high_level_action_budget`，计为失败） | 16 | SwingXtimes kb_g 5、StopCube kb_m 5、StopCube ld_m 4、RouteStick ld_g 1、SwingXtimes kb_m 1 |
| 30 min episode 超时 | 0 | 最长 episode 388 s |
| 300 s 执行器超时 | 0 | — |
| 无效回答（api_error） | 0 | 1,789 / 1,789 次调用 valid |
| 基础设施错误 / incomplete | 0 | — |

**没有走到的四条路径**在 G1 里无法检查；watchdog 的这些分支只在 v2.1 的 watchdog selftest 里测过
（`paper/ROBOMME_WATCHDOG_V2_REPORT_ZH.md`）。

**StopCube 的 50 次上限是怎么来的（看过日志）：** kb_m episode 0 从第 10 次决策起，观测时间停在 t = 393，方块已经
停下；「remain static」不再产生新帧，模型反复选 B，直到 50 次。ld_m 同样。这是任务动态（错过时机后方块停住），
不是记忆或 harness 的 bug；但它说明 StopCube 上「错过即卡死」会消耗满 50 次调用。

## 4. 公平检查（真实请求）

- 策略文本：每个 (任务, 策略) 在 fh / ld / mm / kb 四臂的每次调用里 `policy_sha256` 只有一个值；P-generic 的
  sha 为 `64e4a1f3…`（orchestrator 裁定后的文本），P-matched 与 manifest 一致。
- 记忆说明：每个记忆臂在 g / m 两种策略下 `memory_note_sha256` 只有一个值。
- 记忆里的建议性用词：G1 前的 probe 审计（`results/memory_or_policy/g1_prep/fairness.json`）对 fh / ld / kb 判失败、
  全部通过；mm 只计数：RouteStick 9、StopCube 6、SwingXtimes 32、VideoRepick 8、VideoUnmaskSwap 9。
- 模型：1,789 次调用全部返回 `openai/gpt-6-luna`；0 次重试失败。
- episode：每个 (任务, episode) 的 9 个臂 reset digest 相同；provenance 全部是 `ac55555`、clean。

## 5. 花费与耗时

| 臂 | 花费 |
|---|---:|
| fh_g / fh_m | $0.0156 / $0.0214 |
| ld_g / ld_m | $0.0368 / $0.0626 |
| mm_g / mm_m | $0.0368 / $0.0250 |
| kb_g / kb_m | $0.0576 / $0.0576 |
| memi | $0.0384 |
| **合计** | **$0.3518** |

各任务平均每格：RouteStick 208 s、VideoRepick 184 s、SwingXtimes 154 s、StopCube 123 s、VideoUnmaskSwap 69 s；
平均 147 s/格。照此推算，正式验证 9 × 250 = 2,250 格在 Mac 上串行约 **92 h（3.8 天）**，费用约 **$3.5**
（每格 $0.0016）。

## 6. 需要注明的几点（orchestrator 要求）

- **逐句标注体系与 NOTES 不同：** 规则 4 的标注改为描述性，主类二选一（choice rule / memory update），memory
  reference 是独立标记，不是第三个互斥类；脚本算字面字段名，人工核对部分名字。各任务比例见
  `paper/G1_PACKAGE_ZH.md` §6。
- **MEMI 策略里的两句 train episode 事实：** RouteStick-09「Initial subgoals decoded from the demonstrations are B for
  episode 0, A for episode 19, D for episode 58, A for episode 86, and A for episode 99.」与 SwingXtimes-09「episode 72's
  first right target is visible earlier in the pickup interval」。prompt 里不出现 episode 编号，对 val / test 应无直接作用，
  但说明 MEMI 的策略部分记住了 train 集。
- **H2 的 MM 注释：** MM-m − LD-m 要注明「MM 的记忆按 MEMI 的设计可能编码了动作」（MEMI 的 fields 含观测→菜单选项的映射）。
- **P-generic** 是 factorial P0 把「the memory snapshot」换成「the memory given to you」（orchestrator 2026-11-01 裁定）。

## 7. 下一步（等 orchestrator / 用户决定）

1. 在 NOTES 写带日期的 n 小节：n = 250，n = 250 时可检测约 16.5 pp，等价形式不主张。
2. 冻结正式验证的 manifest（test 0–49 × 5 任务 × 9 臂，一次读取，记 `results/test_reads.log`），包括预注册的
   「基础设施错误最多重跑 2 次、否则所有臂丢掉这一对」——当前 runner 遇到第一个基础设施错误就停，正式验证前要实现。
3. 正式验证约 92 h（Mac 串行）或上 pod（并行宽度未测）。
