# RoboMME Memory × Policy 因子试验报告

状态：**已完成，探索性 pilot**

分支：`research/perception-belief-robomme`

冻结代码提交：`4df3d62`

运行：`pilot_20261101T0453Z`

manifest：`3a5d0118ab936c87`

模型：`openai/gpt-6-luna`（OpenRouter 返回模型 ID 精确匹配）

## 结论摘要

本试验完整执行了预注册的 24 个单元，没有选择性重跑。全部单元均得到协议认可的终态：16 个成功、
8 个任务失败、0 个 timeout、0 个 infrastructure error、0 个 incomplete。公平性、泄漏、模型、重置、
种子、prompt hash、预算与 provenance 审计全部通过。因此，**实验执行完整性为 GO**。

研究结果不支持扩大以完整 ArbiterKB 为中心的 validation。主要比较 D−B（相同 matched policy 下，
ArbiterKB 减 deterministic ledger）为 **−16.7 个百分点**：4/6 对 5/6，paired bootstrap 95% interval
为 [−66.7, +33.3] 个百分点，exact McNemar p=1.0。区间很宽；本试验既不能证明 ArbiterKB 更差，
也不能证明二者等价。但目前没有观察到完整概率 KB 的总体优势，而最简单的 matched ledger 是四臂中
成功率最高者。因此，**当前研究 gate 为 NO-GO：不进入更大 KB-centered validation，不访问 held-out，
不运行 observation degradation，也不扩张 KB rules。**

结果也揭示了一个更具体的问题：完整 KB 的 controller view 同时暴露 normalized execution-only counts
和包含 demo 的 legacy `Progress`。在 VideoRepick episode 1/2，前者是 2 次而后者显示 3 次，模型在两种
policy 下均过早按下按钮。这不是 identity failure，而是重复且冲突的 policy state。另一方面，
VideoUnmaskSwap episode 1 的 ledger 已保存正确 cover relation，却因语义展示较弱而选择错误 container；
Arbiter view 的明确自然语言绑定帮助两臂成功。这个局部优势是**语义呈现优势**，不是概率、校准或
retraction 的因果证据。

## 1. 研究范围与限制

试验严格使用 2×2 设计：

| 条件 | Memory | Policy | 简写 |
|---|---|---|---|
| A | deterministic ledger | generic | `ledger_generic` |
| B | deterministic ledger | task-matched | `ledger_matched` |
| C | frozen ArbiterKB | generic | `arbiter_generic` |
| D | frozen ArbiterKB | task-matched | `arbiter_matched` |

任务固定为 `VideoRepick` 与 `VideoUnmaskSwap`，每任务 3 个成对 episode（0、1、2），共 24 cells。
所有 validation episode 在此前工作中均已使用；为避免接触 held-out，本 pilot 复用 0–2。因此这些结果
不是独立验证，只能生成假设，不能用于 superiority、equivalence 或正式机器人系统成功率结论。

没有运行 held-out test，没有运行 observation degradation，没有改变 ArbiterKB rules，也没有在看过
结果后改 prompt、episode、seed、budget 或选择性重跑。MEMI 未纳入本试验，因为其 memory 与 task policy
联合优化，不能作为 memory-only baseline。

## 2. 技术 gate 与实验完整性

### 2.1 运行前 gate

- 分支和 `main` 校验通过；`main` 保持 `069beb7`。
- 代码、manifest、episodes、seeds、prompt bytes、统计方法和费用上限在 scored calls 前提交冻结。
- exact Luna endpoint probe 通过；返回 `openai/gpt-6-luna`，没有 fallback。
- 两个任务的 simulator/executor preflight 均得到合法终态。
- unit、watchdog、quality 与完整测试套件在运行前通过。
- authorized credential 仅由 macOS Keychain 注入；secret 未写盘、未打印、未提交。

第一次 probe 因 JSON-mode 请求文字未包含单词 `JSON` 而收到 HTTP 400。该 probe 在任何 scored rollout
之前失败，原始失败记录被保留。只修正 probe 的请求文字、重新提交后再次 probe；没有改变 experiment
prompt 或 scored code。成功 probe 后才执行 frozen pilot。

### 2.2 运行后完整性审计

| 检查 | 结果 |
|---|---|
| 24/24 cells 有协议认可终态 | 通过 |
| success / task_failure | 16 / 8 |
| timeout / infrastructure / incomplete | 0 / 0 / 0 |
| 同 task-episode 的 reset digest 与 episode resolution | 一致 |
| 同 policy 条件跨 memory 的 system/policy SHA | 一致 |
| requested/returned model | 精确 `openai/gpt-6-luna` |
| invalid/error model calls | 0 |
| executor API errors | 0 |
| dirty source provenance | 0 |
| simulator actor identity leakage | 未发现 |
| task-progress oracle | 未使用 |
| held-out / degradation | 未访问 / 未运行 |
| selective rerun | 无 |

这里的 audit 证明的是协议执行可信，不证明 memory intervention 在语义信息量上完全等价。事实上，
失败分析显示两种 memory 的自然语言 controller view 仍有不同的冗余和显式语义，这正是本 pilot 发现的
重要接口混杂因素。

## 3. 成功率

括号为 Wilson 95% interval。每任务 n=3、pooled 每臂 n=6，区间很宽。

| Arm | VideoRepick | VideoUnmaskSwap | Pooled |
|---|---:|---:|---:|
| A ledger_generic | 2/3 = 66.7% [20.8, 93.9] | 2/3 = 66.7% [20.8, 93.9] | 4/6 = 66.7% [30.0, 90.3] |
| B ledger_matched | 3/3 = 100% [43.9, 100] | 2/3 = 66.7% [20.8, 93.9] | 5/6 = 83.3% [43.6, 97.0] |
| C arbiter_generic | 0/3 = 0% [0, 56.2] | 3/3 = 100% [43.9, 100] | 3/6 = 50.0% [18.8, 81.2] |
| D arbiter_matched | 1/3 = 33.3% [6.1, 79.2] | 3/3 = 100% [43.9, 100] | 4/6 = 66.7% [30.0, 90.3] |

### 3.1 每 episode 结果与动作轨迹

`S` 为 success，`F` 为 task failure；动作序列按 RoboMME multi-choice 字母记录。

| Task / episode | A ledger_generic | B ledger_matched | C arbiter_generic | D arbiter_matched |
|---|---|---|---|---|
| VideoRepick / 0 | F `A` | S `ABC` | F `ABA` | S `ABC` |
| VideoRepick / 1 | S `ABABABC` | S `ABABABC` | F `ABABC` | F `ABABC` |
| VideoRepick / 2 | S `ABABABC` | S `ABABABC` | F `ABABC` | F `ABABC` |
| VideoUnmaskSwap / 0 | S `A` | S `A` | S `A` | S `A` |
| VideoUnmaskSwap / 1 | F `ABA` | F `ABA` | S `ABA` | S `ABA` |
| VideoUnmaskSwap / 2 | S `A` | S `A` | S `A` | S `A` |

没有连续重复动作 loop；没有 invalid action。相同字母序列不一定代表相同 target，VideoUnmaskSwap / 1
的第三个 `A` 正是 memory view 导致的 target 差异。

## 4. 预注册效应

### 4.1 主要比较：D−B

| 范围 | D | B | Paired RD | Bootstrap 95% interval | D-only / B-only | exact p |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 4/6 | 5/6 | −16.7 pp | [−66.7, +33.3] pp | 1 / 2 | 1.0 |
| VideoRepick | 1/3 | 3/3 | −66.7 pp | [−100, 0] pp | 0 / 2 | 0.5 |
| VideoUnmaskSwap | 3/3 | 2/3 | +33.3 pp | [0, +100] pp | 1 / 0 | 1.0 |

Pooled Holm-adjusted p=1.0。任务方向相反：Arbiter 在 identity-demanding episode 上观察到一次局部收益，
却在 progress-demanding task 上出现两次稳定的提前终止。小样本不允许把任何一边解释成真实平均效应。

### 4.2 次要比较与因子效应

| Contrast / effect | Estimate | Paired bootstrap 95% interval | exact p | Holm p |
|---|---:|---:|---:|---:|
| B−A：matched policy on ledger | +16.7 pp | [0, +50.0] | 1.0 | 1.0 |
| D−C：matched policy on Arbiter | +16.7 pp | [0, +50.0] | 1.0 | 1.0 |
| C−A：Arbiter vs ledger under generic | −16.7 pp | [−66.7, +33.3] | 1.0 | 1.0 |
| Memory main effect | −16.7 pp | [−66.7, +33.3] | 描述性 | — |
| Policy main effect | +16.7 pp | [0, +50.0] | 描述性 | — |
| Memory × policy interaction | 0 pp | [0, 0] | 描述性 | — |

Interaction 的 [0,0] 只是这 6 个冻结 block 中 B−A 与 D−C 恰好逐项形成相同平均差，不代表总体交互
被精确估计为零。

## 5. 失败分析

### 5.1 VideoRepick / 0

- A `ledger_generic`：第一步选择了像素 `[100,160]` 的错误 cube；演示目标实际是 `cube_1`，位于约
  `[78,117]`。ledger 的 normalized target 正确，但 generic policy 没有遵守，属于 target-selection /
  policy adherence error。
- C `arbiter_generic`：正确选择目标后执行 `A,B,A`，而 N=1，最后多做一次 pickup 导致失败，属于
  termination/progress policy error。
- B、D 的 matched policy 均执行 `A,B,C` 并成功，说明该 episode 的策略修复有效。

### 5.2 VideoRepick / 1 与 / 2

C 和 D 在两个 episode 上都执行 `A,B,A,B,C`，只完成两轮就按按钮，而要求是三轮。LLM 调用前的
normalized state 明确显示 execution-only `pick-ups=2, releases=2`；但 Arbiter 原有 controller view
同时显示 `Progress: pick-ups 3, releases 3`，其中包含 demonstration event。两种 matched prompt 的字节
完全相同，D 仍被冲突信息诱导提前终止。

这四次失败应归类为 **state presentation / progress-count conflict**，不是 object identity、executor 或
watchdog error。完整 KB 输出的冗余在这里是负担。B 在同样 matched policy 下用 `ABABABC` 完成 3/3，
也证明 simulator、executor 和任务本身可完成。

### 5.3 VideoUnmaskSwap / 1

A、B 都先正确选择红 cube 对应的 `bin_1`（约 `[95,118]`），放下后第三步又选择同一个位置，未选择
蓝 cube 对应的 `bin_2`（约 `[74,158]`），因此失败。ledger 的机器可读 belief 从一开始就是正确的：
`cube_red→bin_1`、`cube_blue→bin_2`；错误不是 tracking relation 丢失，而是 LLM 没有从简洁 ledger view
稳定执行第二个语义目标。

C、D 的 Arbiter belief 也指向同样 container（约 0.99），但其 joined facts 明确写出 “blue cube task item
under bin_2”，两臂均选择 `[74,158]` 并成功。这是一个真实的 observed view advantage；然而两个 memory
都已知道正确 relation，且没有发生需要概率校准或 retraction 的歧义，所以不能把该差异归因于概率 KB。

### 5.4 失败类别汇总

| 类别 | Cells | 数量 |
|---|---|---:|
| target selection / generic policy adherence | VideoRepick/0 A | 1 |
| task-progress / termination | VideoRepick/0 C | 1 |
| conflicting duplicate progress presentation | VideoRepick/1–2 C、D | 4 |
| semantic goal-binding / view presentation | VideoUnmaskSwap/1 A、B | 2 |
| identity belief actually incorrect | — | 0 |
| executor / invalid action / loop / timeout / infrastructure | — | 0 |

## 6. 成本、tokens 与时间

总 API 费用为 **$0.01536238**：scored calls $0.01535708，成功 probe $0.00000530，远低于 $0.10 cap。
失败 probe 没有成功调用费用。四臂共 78 次模型调用。

| Arm | Calls | Input tokens | Output tokens | Cost | Reset sum / mean | Planner sum / mean | Executor sum / mean | Episode total sum / mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A ledger_generic | 20 | 42,477 | 1,957 | $0.00370286 | 383.6 / 63.9s | 55.0 / 9.2s | 447.8 / 74.6s | 887.8 / 148.0s |
| B ledger_matched | 22 | 47,905 | 1,486 | $0.00369480 | 386.6 / 64.4s | 45.9 / 7.7s | 500.3 / 83.4s | 934.6 / 155.8s |
| C arbiter_generic | 18 | 46,427 | 1,192 | $0.00393298 | 384.7 / 64.1s | 42.9 / 7.2s | 416.5 / 69.4s | 846.0 / 141.0s |
| D arbiter_matched | 18 | 47,602 | 1,289 | $0.00402644 | 378.9 / 63.1s | 41.7 / 7.0s | 432.7 / 72.1s | 855.2 / 142.5s |

全试验 episode wall time 合计 3523.6s（58m43.6s）；阶段累计 reset 1533.8s、planner/LLM 185.6s、
executor 1797.2s。阶段和与总时间可能有少量事件记录边界差异。API 费用差异很小，当前决策主要由成功率、
可解释性与复杂度，而不是费用驱动。

## 7. Go / No-Go

### 执行 gate：GO

24/24 cells 完整、合法、可审计；没有系统性 harness error。可以把结果作为一个可信的探索性 pilot
进行分析和公开负结果记录。

### 研究 gate：NO-GO

不应立即扩大完整 ArbiterKB validation。原因是：

1. 主要比较没有观察到 ArbiterKB 总体优势，点估计反而为 −16.7 pp；
2. matched ledger 以 5/6 成为最佳臂，且结构明显更简单；
3. Arbiter 的 4 个失败来自其重复/冲突的 progress view，而不是 ledger 无法表达的 world state；
4. Arbiter 在 Unmask 的局部优势可以由语义呈现解释，无需概率、校准或 retraction；
5. 样本复用且 n=6，证据不足以支持更昂贵的 KB-centered validation 或 held-out gate。

建议暂时采用**最简 deterministic semantic ledger**作为默认研究基线，冻结完整 KB，不删除历史实现，
但停止增加规则。只有下一项更强的隔离实验出现预注册的、可重复的 uncertainty-specific 收益，才重启
完整 KB 路线。

## 8. 五个问题的直接回答

### 1. 相同 task policy 下，ArbiterKB 是否观察到优于 deterministic ledger？

**总体没有。** D=4/6、B=5/6，D−B=−16.7 pp，95% paired bootstrap interval [−66.7,+33.3] pp。
VideoUnmaskSwap 有一次 D-only 成功，但 VideoRepick 有两次 B-only 成功；局部正例不足以构成概率 KB 优势。

### 2. 先前 VideoRepick 失败有多少可归因于 task-policy mismatch？

当前复用的 3 个 episode 中，matched policy 令 ledger 从 2/3 到 3/3、Arbiter 从 0/3 到 1/3，即观察到
**+33.3 pp 的 policy effect**；但不能把这个小样本比例外推为先前 0/9 的精确归因比例。结合先前 forensics，
0/9 的轨迹都呈 progress/termination mismatch，而非 identity/executor failure；本 pilot 进一步表明即使加入
matched instructions，Arbiter 的 legacy demo-inclusive progress 仍与 normalized counts 冲突。所以合理结论是：
先前失败主要是 policy/interface mismatch，但并非只靠追加 prompt 就已全部修复。

### 3. 当前完整 KB 的复杂度是否有证据支持？

**没有。** 在本任务集合与当前接口上，简单 ledger 更高的 pooled success、较低结构复杂度、正确的 cover
relation，以及完整 KB 自身制造的 progress conflict，都不支持继续扩张 KB。不能据此证明完整 KB 永远无用，
但当前 burden of proof 在 ArbiterKB 一侧。

### 4. 单一最有价值的下一项实验是什么？

进行一次**fresh、冻结、充分把握的 memory-only validation**：两种 memory 都输出相同的 canonical semantic
state，由同一个 deterministic controller/FSM 消费，不再让 LLM 解释不同风格的 memory prose；比较最简
deterministic semantic ledger 与 frozen Arbiter belief state，任务只选择真正需要 identity uncertainty /
retraction 的 episode。episode 来自新预留的 development/validation partition，不复用本 pilot，也不先触碰
held-out。这样能一次性去掉 policy adherence、语义措辞和重复 progress fields 三个主要混杂因素。

### 5. 什么结果会证伪当前假设？

当前假设是：在这些任务上，简单 deterministic semantic ledger 足够，完整概率 KB 没有可辩护的增益。
证伪标准应预注册为：在上述 fresh、leak-free、同 canonical state schema、同 deterministic controller 的
adequately powered paired validation 中，ArbiterKB 相对 ledger 的成功率提升至少 **10 pp**，区间排除 0，
并且成功差异的轨迹可明确归因于概率不确定性、证据更新或 retraction，而不是额外语义标签、prompt 长度、
progress 计数或信息泄漏。若达到该标准，才有理由进入 held-out test。

## 9. 可复现性与产物

- 冻结协议：`paper/ROBOMME_MEMORY_POLICY_FACTORIAL_PROTOCOL_ZH.md`
- 冻结 manifest：`experiments/robomme_memory_policy_factorial_v1.json`
- 机器汇总：`analysis/summary.json`
- 每 cell 轨迹：`analysis/per_cell_traces.json`
- 失败归因：`analysis/failure_diagnostics.json`
- 公平性/泄漏审计：`analysis/fairness_leakage_audit.json`
- timing/cost：`analysis/timing_cost_breakdown.json`
- provenance 与 hashes：`analysis/reproducibility_manifest.json`
- 原始 `result.json` 与 `events.jsonl`：全部 24 cells 原样保留

执行镜像为 `arbiterkb-robomme-cpu:bb6c37c`，本地 image ID 为
`sha256:07424f3b53e08c0bfd27db3a97d89210de92679b040aed3a5aa49e580e059fd1`。pinned sources 与原始结果
SHA-256 均记录在 reproducibility manifest。
