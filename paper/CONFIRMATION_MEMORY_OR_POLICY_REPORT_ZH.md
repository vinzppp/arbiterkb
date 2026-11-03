# 「Memory or policy?」正式验证报告

日期：2026-11-03

运行：`confirmation_20261102T170540Z`

分支：`research/perception-belief-robomme`

运行代码：`797894d86cd7559d494b9535fccb5623302b27be`（所有 cell 均为 clean）

冻结 manifest：`experiments/memory_or_policy_confirmation_v1.json`，SHA `67df681ec8d4ee4d`

机器可读报告：`results/robomme_common/memory_or_policy_confirmation/confirmation_20261102T170540Z/confirmation_report.json`

## 结论先行

1. **H1（policy 主效应）得到支持。** 四种 memory 的 matched policy 相对 generic policy 的平均增益为
   **+19.6 pp**，task-stratified bootstrap 95% CI **[+16.1, +23.1] pp**；双侧 sign-flip
   `p = 0.00010`，P1/P3 两项 Holm 校正后 `p = 0.00020`。区间下界高于预注册的 +10 pp 实用阈值。
2. **H2 不作等价性结论。** 按 G1 后、读取 test 前冻结的决定，P2a–P2c 只报告区间；n=250 在 G1
   discordance 下约只能检测 16.5 pp。结果分别是 MM-m − LD-m **+32.8 pp**、KB-m − LD-m
   **−13.2 pp**、FH-m − LD-m **+2.0 pp**。最后一个接近零也不能被解释为等价。
3. **H3（FullHist policy gain 的可移植性）被证伪。** 预注册统计量
   `(FH-m − FH-g) − 0.5 × (MEMI − FH-g)` 为 **−8.6 pp**，95% CI **[−13.4, −3.8] pp**；
   双侧 sign-flip 与 Holm 校正后均为 `p = 0.00310`。FH matched policy 的增益为 +18.8 pp，而
   stock MEMI 相对 FH-g 的增益为 +54.8 pp，点估计恢复比例只有 **0.343**。因此 matched policy
   没有恢复预注册要求的至少一半 MEMI 优势。
4. **运行完整且可计分。** 2,250/2,250 cells、250/250 `(task, episode)` pairs；0 infrastructure
   error、0 failed retry、0 pair drop、0 provenance issue。六个 shard parent 均以 0 退出。总模型费用
   **$3.55997682**，未触发 `$6` stop。

## 1. 设计与统计口径

- held-out test 0–49，5 个 task，9 个 arms，共 250 pairs / 2,250 cells；每个 pair 的九臂使用同一
  resolved episode、seed、difficulty 与非空 reset digest。
- 模型为 `openai/gpt-6-luna`，effort `low`；每 cell 最多 50 次 high-level action。
- success、task failure 与 timeout 均按 intention-to-treat 计分；只有预注册的 infrastructure error
  才允许初次之后最多两次相同重跑，仍失败则九臂共同排除。本次没有触发重跑或排除。
- P1 和 P3 是唯一的 Holm family；各自使用 10,000 次、seed 0 的双侧 sign-flip test。区间使用按
  task 分层、每 task 内重采样 pair 的 10,000 次 bootstrap（seed 0）。P2 只报告区间。

## 2. 每臂成功数

每个 task 每臂分母为 50；合计分母为 250。

| 臂 | RouteStick | SwingXtimes | VideoUnmaskSwap | StopCube | VideoRepick | 合计 | 成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fh_g | 1 | 12 | 16 | 29 | 18 | 76 | 30.4% |
| fh_m | 4 | 35 | 18 | 46 | 20 | 123 | 49.2% |
| ld_g | 2 | 8 | 41 | 25 | 28 | 104 | 41.6% |
| ld_m | 3 | 34 | 39 | 2 | 40 | 118 | 47.2% |
| mm_g | 28 | 13 | 32 | 2 | 19 | 94 | 37.6% |
| mm_m | 31 | 49 | 38 | 45 | 37 | 200 | 80.0% |
| kb_g | 1 | 0 | 36 | 0 | 19 | 56 | 22.4% |
| kb_m | 2 | 22 | 35 | 2 | 24 | 85 | 34.0% |
| memi | 45 | 47 | 33 | 47 | 41 | 213 | 85.2% |

单臂成功率是描述性结果；主检验使用同 pair 的配对差值，而不是把这些边际比例当作独立样本。

## 3. 主检验

| 检验 | 点估计 | task-stratified 95% CI | raw p | Holm p | 预注册判读 |
|---|---:|---:|---:|---:|---|
| P1：四种 memory 上 matched − generic 的平均值 | +19.6 pp | [+16.1, +23.1] | 0.00010 | 0.00020 | **H1 supported** |
| P3：`(FH-m−FH-g) − 0.5×(MEMI−FH-g)` | −8.6 pp | [−13.4, −3.8] | 0.00310 | 0.00310 | **H3 falsified** |

P3 的组成量为：FH-m − FH-g **+18.8 pp** [13.2, 24.4]；MEMI − FH-g **+54.8 pp**
[48.4, 60.8]。分母区间排除 0，因此报告恢复比例 0.343。

P3 的任务分解显示强异质性：

| task | P3 统计量 | 95% CI |
|---|---:|---:|
| RouteStick | −38 pp | [−45, −30] |
| SwingXtimes | +11 pp | [−2, +24] |
| VideoUnmaskSwap | −13 pp | [−22, −4] |
| StopCube | +16 pp | [+4, +27] |
| VideoRepick | −19 pp | [−31, −7] |

整体结论按预注册的 task-balanced 统计量给出；任务分解不另作多重检验后的确认性主张。

## 4. P2 次要区间与 split fidelity

| 对比 | 点估计 | 95% CI | 前者独胜 | 后者独胜 | discordant / 250 |
|---|---:|---:|---:|---:|---:|
| P2a MM-m − LD-m | +32.8 pp | [+26.8, +38.4] | 98 | 16 | 114 |
| P2b KB-m − LD-m | −13.2 pp | [−18.8, −7.6] | 13 | 46 | 59 |
| P2c FH-m − LD-m | +2.0 pp | [−4.0, +8.0] | 61 | 56 | 117 |
| split fidelity MM-m − stock MEMI | −5.2 pp | [−10.4, 0.0] | 19 | 32 | 51 |

P2a 的 MM memory 按 MEMI 设计可能编码动作（fields 包含 observation 到 menu option 的映射），所以不能把
MM-m 相对 LD-m 的差值解释为纯粹的状态表示收益。P2c 的区间比 ±5 pp 等价界更宽，且本研究按冻结决定没有进行
TOST；**不能声称 FH-m 与 LD-m 等价**。

split fidelity 的点估计绝对值为 5.2 pp，没有超过预注册的 10 pp caveat 触发线；不过区间下端为
−10.4 pp，应保留不确定性。最终 point estimate 不支持“替换实现本身造成超过 10 pp 偏差”的说法。

## 5. 终止路径、成本与时间

| 终态 / 原因 | 数量 | 计分方式 |
|---|---:|---|
| success / `environment_success` | 1,069 | 成功 |
| task failure / `fail` | 1,018 | 失败 |
| task failure / `planner_error` | 25 | 失败 |
| timeout / `high_level_action_budget` | 138 | 失败（50-call cap） |
| infrastructure error / incomplete | 0 | 本次无排除 |

这里的 138 个 `timeout` 全部是 50-call high-level-action cap，不是 30 分钟 episode wall timeout，也不是
executor timeout。`planner_error` 按冻结的 ITT 规则属于 policy/task failure，不允许重跑。

- 各 cell `seconds` 合计 92,514.4 s（25 h 41 min 54.4 s）；六分片实际墙钟约 25 h 29 min 27 s，
  从 `2026-11-02T17:05:49Z` 到最后结果 `2026-11-03T18:35:16Z`。
- 总模型费用 `$3.55997682`。按臂：FH-g/m `$0.20085/$0.20762`，LD-g/m
  `$0.34316/$0.56939`，MM-g/m `$0.39652/$0.29345`，KB-g/m `$0.52839/$0.56542`，
  MEMI `$0.45519`。
- 总 high-level model calls：FH-g/m 1,024/1,241，LD-g/m 1,747/3,198，MM-g/m
  1,347/1,084，KB-g/m 3,386/3,502，MEMI 1,131。

## 6. StopCube 与解释边界

正式验证前的免费审计已经确认，官方 MEMI executor 对 on-menu `remain static` 的 zero-batch 结果记录为
`no_op`，不推进 simulator time；本 harness 与它一致。证据与固定源码位置见
`paper/STOPCUBE_EXECUTOR_AUDIT.json`。因此 G1 中 t=393 后反复选择 `remain static` 不是 harness divergence；
更准确的表述是官方 no-op 语义与 policy 选择的交互，而不是仅凭现象归因于 task dynamics。

StopCube 的臂间差异很大（例如 LD-m 与 KB-m 各 2/50，FH-m 46/50，MM-m 45/50，MEMI 47/50），说明
matched policy 的收益取决于它和 memory 表示的组合；这与 P3 的跨任务异质性一致，但不改变预注册的整体判读。

## 7. 完整性与 provenance

- 2,250 个 `result.json` 和 2,250 个 `events.jsonl`；原始结果/事件树 SHA-256：
  `10a06c16444cc8f23ff2b56b1a4d6fa5dae6d7299d62cd17492733a9b6b37cd3`。
- manifest `67df681ec8d4ee4d`；ArbiterKB
  `797894d86cd7559d494b9535fccb5623302b27be`，全部 `arbiter_dirty=false`。
- MEMI `bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0`；RoboMME
  `fae9f7300347f1146c2ae4844faa4eef7e38098e`；ManiSkill
  `07be6fbc66350ddca200abfb0a11b692f078f7fd`。
- params `fe4e94112a74`；prompt `307eaf7ff479aeae`；请求与返回模型均核对为
  `openai/gpt-6-luna`，effort `low`。
- RunPod pod `3ics2knpm77sov`，NVIDIA RTX PRO 4500 Blackwell 32 GB；六 shards 按 pair 分片。
- 本次读取已以 manifest SHA 写入 `results/test_reads.log`；没有在报告阶段重新运行 episode 或调用模型。

机器报告重新生成命令：

```bash
.venv/bin/python scripts/confirmation_report.py \
  results/robomme_common/memory_or_policy_confirmation/confirmation_20261102T170540Z \
  --manifest experiments/memory_or_policy_confirmation_v1.json \
  --out results/robomme_common/memory_or_policy_confirmation/confirmation_20261102T170540Z/confirmation_report.json
```

## 8. 解释限制

- 五个 task 是根据已发布验证工作中的差距选择的，不是从所有 RoboMME task 随机抽样；结论直接适用于这五个
  held-out task，而不是无条件外推到整个 benchmark。
- 相同模型、固定 matched policy 与固定 memory 实现是本次估计对象；“policy 比 memory 更重要”应理解为本设计下
  的平均因果对比，不等于 memory 在所有 task 或 policy 下都不重要。
- 任务级 H3 结果异质；总体 H3 falsification 表示预注册的“至少恢复一半 MEMI 优势”没有成立，不表示 matched
  policy 在每个 task 都无益。
