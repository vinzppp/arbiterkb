# ArbiterKB R2.1：RoboMME 冻结 validation 失败取证

日期：2026-10-31（America/Los_Angeles）

分支：`research/perception-belief-robomme`

冻结 run：`validation_20261031T150046Z`
分析性质：既有 validation 的离线、事后机制诊断；不是新的确认性实验

## 1. 执行摘要

本报告没有调用模型、没有启动模拟器、没有重跑任何 cell，也没有读取 held-out 或开始
observation degradation。输入是冻结 run 的 144 个 `result.json`、144 个 `events.jsonl`、
`summary.json` 和实际运行代码/提示词。派生表由
`scripts/robomme_failure_forensics.py` 从原始结果重算；每个源文件的 SHA-256 都保存在
`failure_forensics.json`，原数据未改动。

结论比“ArbiterKB 在 VideoRepick 上表现不好”更具体：

1. **ArbiterKB 的 0/9 不是执行器或 watchdog 失败。** 9/9 首次 `A` 都被环境接受，
   35/35 决策全部执行；没有 API error、planner error、executor rejection、loop 或 timeout。
2. **9/9 的可见失效都发生在任务进度/终止策略。** 对 `N=3` 的四个 episode，实际序列均为
   `ABABC`，比正确 `ABABABC` 少一个执行 cycle；对 `N=2` 的两个 episode，实际序列均为
   `ABC`，比正确 `ABABC` 少一个 cycle；对 “again” 的三个 `N=1` episode，实际均为
   `ABA`，本应在 `AB` 后执行 `C`。
3. **KB 的物理事件记录大体正确，但任务语义错位。** 初始 snapshot 已把 demonstration 的一次
   grasp/release 记为 `pick-ups 1, releases 1`。common harness 没有把已有的 goal counter 注入
   ArbiterKB，所以 planner 只看到 demonstration 与 execution 混合的累计数，而不是“execution
   cycles 0 of N”。在 `N=2/3` 失败中，这正好造成提前一个 cycle 结束。`N=1` 的错误方向不同；
   trace 只能支持“缺少明确的 task policy / termination semantics”，不能证明唯一内部原因。
4. **MEMI 的优势与 memory 因果效应混杂。** MEMI 的 VideoRepick prompt 明确规定
   `completed_cycles=0`、忽略 demonstration、解析 `again/two/three`、并用近似 FSM 的规则选择
   `A/B/C`。ArbiterKB 没有等价规则。当前四臂实验比较的是“任务专用 memory + 任务专用 policy”
   对“通用 KB + 通用 policy”，不能识别 memory、update rule 和 task policy 的独立贡献。
5. **整体结论仍然是负面。** ArbiterKB 15/36，低于 FullHist 17/36 和 MEMI 24/36。六个冻结
   总体对比中，只有 MEMI > soft 在 Holm 校正后显著。ArbiterKB 的复杂 KB 在这组 clean、
   simulator-semantic validation 上没有证明比简单 history、deterministic ledger 或直接 FSM 更有必要。

研究建议：**不要把“修好 VideoRepick counter”当论文方向。** 最有价值的下一实验是一个冻结的
`memory × policy` 2×2 factorial，并加入 deterministic cover/action ledger 与 direct demo-to-FSM
baseline。若 task-specific policy + 简单 ledger 已达到 MEMI，而 KB 的主效应和交互效应接近零或为负，
应停止当前 ArbiterKB 架构；只有在该 falsifier 之后，才值得研究概率 belief/retraction 是否能在
**测得的感知错误**下改变闭环决策。

## 2. 证据范围与完整性

冻结 manifest 为 `a839ccc592f8aa55`；model、executor、reset、seed 和 clean observation 在四臂间匹配。
144/144 cells 有协议允许的终态，65 success、74 task failure、5 timeout；36 个 task/episode 组均有四臂，
reset digest 与 episode seed 跨臂一致。没有 dirty provenance、wrong model、invalid call、API error、
infrastructure error、incomplete 或 harness error。总成本 `$0.13371252`，577 次 model call；五个 episode
wall timeout 原样保留为失败。完整审计见 `reproducibility_manifest.json`。

本报告把三类陈述分开：

- **观察事实**：原始 command、point、environment status、memory snapshot、terminal status；
- **trace 支持的诊断**：例如首抓被环境接受后，choice 在第 4 步从预期 `A` 变成 `C`；
- **机制推断**：例如 demonstration count 进入通用进度行是提前终止的最简解释。若 reasoning 为空，
  不把该推断写成模型的“想法”。

## 3. VideoRepick：正确目标与判定规则

菜单在九个 episode 都是：`A` 抓 cube、`B` 放下、`C` 按钮结束。目标中的次数指 **execution**
cycle，不包括 demonstration：

| goal 文字 | N | 正确 choice 序列 | 冻结结果中的成功实例 |
|---|---:|---|---|
| `three times` | 3 | `ABABABC` | MEMI ep1/3/8 |
| `two times` | 2 | `ABABC` | FullHist 与 MEMI ep7 |
| `again` | 1 | `ABC` | FullHist ep4；MEMI ep6 |

这既由成功轨迹验证，也与 pinned MEMI VideoRepick prompt 的明确规则一致。`A` 的 point 还必须指向
demonstration 中被抓起并在交换后追踪到的同一 cube。错误目标在这些轨迹中通常会让环境在首个 `A`
立即返回 `fail`；因此本报告把首个 `A` 后的 `ongoing` 记为“环境接受的 target selection”。这是一项
操作性证据，不声称能读取 simulator 内部的 latent identity。

### 3.1 36 条轨迹的逐 episode 诊断

`S/F` 是最终 success/failure；括号内是首个 `A` point。`A×` 表示首抓后环境立即 `fail`；
`A✓` 表示首抓后为 `ongoing`。每行的原始证据位于
`validation/clean/VideoRepick_ep<E>_<arm>/{result.json,events.jsonl}`。

| ep | N / 正确序列 | FullHist | MEMI | Soft repair | ArbiterKB | 配对诊断 |
|---:|---|---|---|---|---|---|
| 1 | 3 / `ABABABC` | `A` F, A× `[95,119]` | `ABABABC` S, A✓ `[99,98]` | `A` F, A× `[96,119]` | `ABABC` F, A✓ `[98,98]` | Arb 与 MEMI 抓同一目标附近；Arb 在 step 4 选 `C`，少一 cycle。其 reasoning 声称“三次已完成”。 |
| 2 | 3 / `ABABABC` | `A` F, A× `[91,98]` | `A` F, A× `[91,98]` | `A` F, A× `[90,99]` | `ABABC` F, A✓ `[91,166]` | 只有 Arb 首抓被接受，但仍在 step 4 提前 `C`；这是 KB-only identity win，却不是 task success。 |
| 3 | 3 / `ABABABC` | `ABABC` F, A✓ `[55,79]` | `ABABABC` S, A✓ `[55,79]` | `A` F, A× `[95,81]` | `ABABC` F, A✓ `[54,79]` | Arb 与 FullHist 都在 step 4 提前 `C`；MEMI 多执行一组 `AB`。Arb reasoning 把累计第三次 release 当完成。 |
| 4 | 1 / `ABC` | `ABC` S, A✓ `[109,155]` | `A` F, A× `[131,119]` | `ABA` F, A✓ `[109,155]` | `ABA` F, A✓ `[109,155]` | FullHist 证明 target 与 `ABC` 可成功；Arb 在 step 2 应 `C` 而选 `A`。无 reasoning，不能再推断。 |
| 5 | 2 / `ABABC` | `A` F, A× `[79,132]` | `A` F, A× `[79,132]` | `A` F, A× `[79,132]` | `ABC` F, A✓ `[112,155]` | 只有 Arb 找到环境接受的目标；其 snapshot 显示累计 2 releases 后选 `C`，但其中 1 次来自 demo。 |
| 6 | 1 / `ABC` | `A` F, A× `[98,95]` | `ABC` S, A✓ `[97,146]` | `A` F, A× `[97,96]` | `ABA` F, A✓ `[97,147]` | Arb 与 MEMI 的 target 相同；MEMI `AB→C`，Arb `AB→A`。Arb reasoning 明说要再抓“第三次”。 |
| 7 | 2 / `ABABC` | `ABABC` S, A✓ `[101,147]` | `ABABC` S, A✓ `[101,147]` | `ABABA` F, A✓ `[101,146]` | `ABC` F, A✓ `[101,147]` | 四臂 target 一致；FullHist/MEMI 正确。Soft 过晚结束，Arb 过早结束，直接隔离出 policy/progress 问题。 |
| 8 | 3 / `ABABABC` | `A` F, A× `[86,98]` | `ABABABC` S, A✓ `[83,156]` | `A` F, A× `[85,99]` | `ABABC` F, A✓ `[83,157]` | Arb 与 MEMI target 一致；Arb 在 step 4 提前 `C`，少一 cycle。 |
| 9 | 1 / `ABC` | `A` F, A× `[97,157]` | `A` F, A× `[97,157]` | `A` F, A× `[96,158]` | `ABA` F, A✓ `[135,126]` | 只有 Arb 首抓被接受；step 2 reasoning 明说要再抓同一 cube，而正确动作是 `C`。 |

逐 arm 汇总：

| arm | 首抓被接受 | 完整成功 | 首个可见分歧 |
|---|---:|---:|---|
| FullHist | 3/9 | 2/9 | 6 次 target/grounding；1 次提前终止（ep3） |
| MEMI | 5/9 | 5/9 | 4 次 target/grounding；被接受后 5/5 成功 |
| Soft | 2/9 | 0/9 | 7 次 target/grounding；2 次终止策略错误（ep4/7） |
| ArbiterKB | **9/9** | **0/9** | **9 次均为 progress/termination choice** |

所以“ArbiterKB 0/9 是 identity memory 崩溃”与原始轨迹矛盾。更准确的说法是：ArbiterKB 的通用
object/event state 在这九个 episode 上找到了环境接受的首抓目标，但没有把任务要求转成正确的
execution-only progress 与终止 policy。它在 identity 子问题上优于其他臂，却在组合任务上全败。
这是 exploratory mechanism analysis，不是新显著性结论。

### 3.2 ArbiterKB memory transition 的具体错位

九个 ArbiterKB episode 的 decision 0 都已经显示 `grasps 1, places 1, pick-ups 1, releases 1`；这些
事件来自 demonstration。之后每个执行 `A/B` 都正确递增通用物理计数。关键不是“memory 被随机破坏”，
而是计数的 reference frame 与任务不一致：

- ep1/2/3/8（N=3）：decision 4 之前为累计 `pick-ups 3, releases 3`，实际只完成 2 个 execution
  cycles；四次都选 `C`。
- ep5/7（N=2）：decision 2 之前为累计 `pick-ups 2, releases 2`，实际只完成 1 个 execution cycle；
  两次都选 `C`。
- ep4/6/9（N=1）：完成 1 个 execution cycle 后累计为 2，却选择另一个 `A`。ep6/9 reasoning 支持
  “还要再抓一次”的错误 policy；ep4 没有 reasoning，因此只报告行为。

代码解释了为什么没有更合适的 line：`ArbiterKBSystem.reset(meta, counter=None)` 支持 goal counter，
但 `DeterministicMemory.reset` 只调用 `system.reset(meta)`，common harness 从未解析并传入 counter。
`controller_view` 因而只能输出 demonstration 与 execution 混合的通用 `Progress:`，不会输出
`Goal progress: ... n of N`。这是 frozen implementation 的事实，不是事后猜测。

“把 counter 接上”很可能修复这一任务，但它只会实现一个 SimpleARM/MEMI/直接 FSM 都已覆盖的
显式 progress state；不能据此主张 KB 的 belief graph、retraction 或规则层有效。

## 4. 全四任务配对错误分析

### 4.1 每 task/episode 成功矩阵

`S`=success，`F`=environment/task failure，`T`=保留为失败的 episode wall timeout。

| task | ep | FullHist | MEMI | Soft | ArbiterKB |
|---|---:|:---:|:---:|:---:|:---:|
| VideoUnmaskSwap | 1 | F | S | F | S |
|  | 2 | S | S | S | S |
|  | 3 | F | S | F | F |
|  | 4 | S | F | S | S |
|  | 5 | S | S | S | S |
|  | 6 | F | S | S | S |
|  | 7 | F | F | F | S |
|  | 8 | F | S | S | S |
|  | 9 | F | S | F | F |
| ButtonUnmaskSwap | 1 | F | S | F | F |
|  | 2 | S | S | F | S |
|  | 3 | F | F | T | F |
|  | 4 | S | S | T | S |
|  | 5 | F | T | T | F |
|  | 6 | F | F | F | F |
|  | 7 | S | F | F | F |
|  | 8 | F | F | T | F |
|  | 9 | F | F | F | F |
| PickHighlight | 1 | S | S | S | S |
|  | 2 | S | S | F | F |
|  | 3 | S | S | F | F |
|  | 4 | S | S | S | S |
|  | 5 | S | S | F | S |
|  | 6 | S | S | F | S |
|  | 7 | S | S | F | F |
|  | 8 | S | S | S | S |
|  | 9 | S | S | S | S |
| VideoRepick | 1 | F | S | F | F |
|  | 2 | F | F | F | F |
|  | 3 | F | S | F | F |
|  | 4 | S | F | F | F |
|  | 5 | F | F | F | F |
|  | 6 | F | S | F | F |
|  | 7 | S | S | F | F |
|  | 8 | F | S | F | F |
|  | 9 | F | F | F | F |

Task totals为：VideoUnmaskSwap `3/7/5/7`，ButtonUnmaskSwap `3/3/0/2`，
PickHighlight `9/9/4/6`，VideoRepick `2/5/0/0`（顺序 FullHist/MEMI/Soft/ArbiterKB）。

### 4.2 冻结六对总体 contrast

所有检验都以 36 个 matched task/episode 为配对单位，不把 144 cells 当独立样本。`A-only/B-only`
是 McNemar discordant cells；RD 的 95% CI 是固定 seed 的 10,000 次 paired episode bootstrap。
Holm family 与原冻结报告相同，包含六个总体 contrast。CI 未作 multiplicity adjustment。

| A vs B | both S | A-only | B-only | both F | RD A−B [95% CI] | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| MEMI vs FullHist | 14 | 10 | 3 | 9 | +0.194 [0.000, +0.361] | .0923 | .1846 |
| Soft vs FullHist | 7 | 2 | 10 | 17 | −0.222 [−0.389, −0.056] | .0386 | .1250 |
| Arbiter vs FullHist | 11 | 4 | 6 | 15 | −0.056 [−0.222, +0.111] | .7539 | .7539 |
| Arbiter vs MEMI | 13 | 2 | 11 | 10 | −0.250 [−0.417, −0.083] | .0225 | .1123 |
| Arbiter vs Soft | 9 | 6 | 0 | 21 | +0.167 [+0.056, +0.306] | .0313 | .1250 |
| MEMI vs Soft | 8 | 16 | 1 | 11 | +0.417 [+0.250, +0.583] | .000275 | **.00165** |

确认性结论仍只有 MEMI > Soft。Arbiter vs MEMI 的未校正 p 与 bootstrap CI 看似负面，但没有通过
冻结的六检验 Holm family；应报告 effect size，不应把它升级为新显著性主张。

MEMI-only vs ArbiterKB 的 11 个 cells 是：ButtonUnmaskSwap ep1；PickHighlight ep2/3/7；
VideoRepick ep1/3/6/8；VideoUnmaskSwap ep1/3/9。Arbiter-only 的两个 cells 是
VideoUnmaskSwap ep4/7；双方都失败 10 个，双方都成功 13 个。这个分布显示 ArbiterKB 的两次独有优势
只在 VideoUnmaskSwap，而劣势跨越 progress、highlight 和 unmask tasks。

### 4.3 Action、终止、loop 与 executor

| arm | policy calls | 已形成 decisions | executed | success / env fail / timeout / budget | loop episodes / events |
|---|---:|---:|---:|---|---|
| FullHist | 88 | 88 | 88 | 17 / 19 / 0 / 0 | 0 / 0 |
| MEMI | 129 | 113 | 113 | 24 / 11 / 1 / 0 | 1 / 1 |
| Soft | 246 | 170 | 170 | 9 / 19 / 4 / 4 | 9 / 15 |
| ArbiterKB | 114 | 114 | 114 | 15 / 21 / 0 / 0 | 0 / 0 |

五个 timeout 都是 process-level episode wall timeout；没有 executor-phase timeout。四个 Soft
`preflight_complete` 是达到冻结 decision budget 后的 task failure。所有 485 个已形成 decision 都有
`result=executed`；没有 `api_error`、`planner_error`、`no_op` 或 `harness_error`。所以失败主要是 planner
选择了错误 action/point 或陷入 non-progress，而不是 low-level executor 拒绝正确指令。

Loop detector 只是 diagnostic，不会改变 action。16 个 loop event 全部出现在 ButtonUnmaskSwap：
Soft 的 9/9 episodes（12 alternating、3 same-action），以及 MEMI ep5 的 1 个 same-action loop；后者最终
timeout。FullHist 与 ArbiterKB 无 loop，VideoRepick 四臂无 loop。

### 4.4 Memory-state accuracy 可测与不可测部分

- 冻结 scorer 只有 VideoUnmaskSwap 的 structured belief ground truth：Soft 与 ArbiterKB 都只有 4 个
  scored rows，均为 4/4。样本过小，且不能外推到 progress 或 target identity。
- VideoRepick 没有逐 step 的显式 identity ground-truth row。环境反馈提供较弱但直接的行为标记：
  ArbiterKB 9/9 首抓被接受，MEMI 5/9、FullHist 3/9、Soft 2/9。
- ArbiterKB 的 physical event count 与执行动作一致；错误是 demonstration/execution 没有分域，故
  execution-only progress 在 9/9 episodes 都未被显式表示。
- MEMI 的成功轨迹中 `completed_cycles` 从 0 开始，每次完整 `B` 后增加，最终到 N；但 MEMI 仍有
  4/9 首抓 identity/grounding failure。这说明其 progress policy 很强，不代表 target memory 完美。

### 4.5 成本与延迟

| arm | mean calls/ep | mean input tokens/ep | mean model latency/ep | mean wall/ep | total estimated API cost |
|---|---:|---:|---:|---:|---:|
| FullHist | 2.44 | 5,746 | 6.32 s | 111.73 s | `$0.01364` |
| MEMI | 3.58 | 10,577 | 16.00 s | 145.41 s | `$0.05637` |
| Soft | 6.83 | 12,120 | 20.42 s | 234.23 s | `$0.03808` |
| ArbiterKB | 3.17 | 7,470 | 10.55 s | 127.40 s | `$0.02563` |

MEMI 比 FullHist 的 paired mean cost 高 `$0.001187/episode`，而成功率差为 +0.194；ArbiterKB 比
FullHist 高 `$0.000333/episode`，成功率差为 −0.056。当前结果不支持 ArbiterKB 的 success、cost 或
latency Pareto 优势。

## 5. Memory 与 task policy 的混杂

### 5.1 MEMI 实际得到的 task-specific 信息

pinned MEMI commit `bb6c37c` 的 `optimized_luna.json` 为每个 task 提供独立 prompt。VideoRepick 条目不只是
“memory schema”，而包含：

- 字段 `target_anchor` 和 `completed_cycles`；
- 如何从 demo grasp、release 和后续 cube exchanges 追踪目标；
- 明确初始化 `completed_cycles=0`，**永不计 demonstration actions 或内部 retries**；
- 明确解析 `again=1, two times=2, three times=3`；
- 明确的 action policy：开始 `A`；holding 时 `B`；release 后若 count < N 则 `A`，否则 `C`；
- 对 gripper posture、release sampling、airborne centroid 和 ambiguous observation 的 task-specific fallback。

这已经接近一个带 perception heuristics 的 FSM。MEMI 每步用 fresh request，但把该 task prompt、旧 memory
和当前 interval 一起送入；模型同时更新 fields 并选择 action。

### 5.2 ArbiterKB 是否得到等价信息

没有。Soft/Arbiter arm 使用 FullHist 的通用 system prompt 加一段通用 `MEMORY-ARM OVERRIDE`；每步 fresh
conversation，输入 goal、menu、当前 segment 与 deterministic snapshot。snapshot 包含 objects、relations、
events、generic progress 和可见 pixel lookup，但没有 VideoRepick 的 N parsing、demo exclusion、release
判定或 `A/B/C` transition rule。

仓库确实有 `goal_counter.py`，但 common harness 没有调用 `parse_goal` 或把 counter 传入
`ArbiterKBSystem.reset`。因此“ArbiterKB 有 L4 counter”不能作为本次运行的解释；本次运行实际没有
task-normalized counter line。

### 5.3 现在能识别什么

支持：

- MEMI package（task schema + task update + task policy）在总体和 VideoRepick 上优于当前通用 Arbiter package；
- VideoRepick 的主要差异是 explicit execution-cycle state 与 termination policy；
- ArbiterKB 的 identity/event state 本身未在这九条轨迹崩溃；
- 当前 full KB 没有超过 FullHist，既有 held-out perception audit 也没有超过 detector-only soft repair。

不可识别：

- MEMI 的增益中多少来自字段、update rule、task policy 或三者 interaction；
- 给 ArbiterKB 同一 task policy 后，KB 是否有增量价值；
- 给简单 ledger 同一 task policy 后，是否已经足够；
- 在真实 perception error 下，ArbiterKB 的 probability/retraction 是否能改变 action 并提高 success。

因此不能把当前 MEMI > ArbiterKB 解释为“learned memory 一定优于 KB”，也不能把 ArbiterKB 的 target
选择表现解释为“KB 已被证明有效”。

## 6. 最小 memory × policy factorial（只设计，不执行）

### 6.1 四个主 cells

使用相同 `openai/gpt-6-luna`、effort、endpoint、clean semantic observations、demo、goal、menu、官方
`multi_choice` executor、decision/token/wall budgets、reset/seeds 和 fresh-call format。唯一两个因素是：

| | P0：通用 action policy | P1：task-specific action/termination policy |
|---|---|---|
| **M0：deterministic neutral ledger** | M0P0 | M0P1 |
| **M1：ArbiterKB structured state** | M1P0 | M1P1 |

M0 是一个不做概率推理的 task-agnostic ledger：demo 与 execution 分区；记录 object track/cover chain、
已执行 command、environment result、execution-only pick/release/press/hold events；不输出下一 action。
M1 保留 ArbiterKB object/belief/retraction state，但必须输出与 M0 完全相同的 execution event fields，防止
“缺字段”伪装成 KB 效应。两种 memory 都不得含 policy recommendation。

P0 是当前通用 goal/menu instruction。P1 是每 task 一份冻结 transition policy；对 VideoRepick 使用
`N` parsing 与 `A/B/C` 规则，但删去 MEMI 的 memory-field/update 指令。P1 的**同一文本**用于 M0 与 M1。
先在非 held-out development/validation episodes 固定，不看 test。

主 estimands 以 task/episode 配对：

- policy 主效应：平均 `P1−P0`；
- KB 主效应：平均 `M1−M0`；
- interaction：`(M1P1−M1P0)−(M0P1−M0P0)`；
- 次要：task-stratified success、action count、loops、terminal cause、cost/latency。

二元 outcome 用 paired cluster bootstrap CI；预注册少量主 contrasts 后 Holm 校正。模型随机性若不可固定，
每个 cell 应有相同的 replicate IDs，并把 replicate 嵌套在 task/episode 中，不能挑最好的一次。

### 6.2 必需的简单 baselines

1. **Direct demo-to-FSM（VideoRepick）**：从 goal 解析 N；用同一 deterministic target tracker 取 anchor；
   只按 execution action result 更新 count；输出 `A→B→...→C`。它不需要 LLM 规划。若它达到接近上限，
   VideoRepick 不应被包装为 KB reasoning 贡献。
2. **Deterministic cover-following ledger**：对 VideoUnmaskSwap/ButtonUnmaskSwap/PickHighlight，以 frozen
   cover/reveal/action ledger 保存目标与 cue，不做 belief graph。它是 ArbiterKB 必须击败的工程下界。
3. **FullHist P0/P1**：可作为 context-rich reference，但不进入最小 2×2 的主 effect。
4. **Stock MEMI**：保留为 bundled-system reference，不纳入 factorial 主效应，因为其 schema/update/policy
   联合优化。

公平性审计必须 diff 实际 prompt，确认四个主 cells 的 policy 文本只随 P 改变、memory payload 只随 M
改变；模型、observation、executor、budget、menu 和 termination 完全一致。

### 6.3 预注册 GO/NO-GO

- **继续研究 KB（GO）**：M1 在 P0 与 P1 下都有方向一致、task-balanced 的增益；KB 主效应 CI 排除
  一个预注册的最小实用差（建议 +10 percentage points），并且不是靠更多 calls/timeouts 换来。
- **只保留 task-state/ledger（NO-GO 当前 KB）**：M0P1 与 M1P1 相当或更好，direct FSM 达到同等 success；
  此时复杂 belief graph 对 clean tasks 没有必要。
- **policy-only 解释成立**：P1 大幅提升两种 memory，而 M1−M0 接近零；MEMI 当前优势主要不能归因于
  structured memory 本身。
- **interaction 值得研究**：只有 M1P1 显著提高，且错误分析证明是 uncertainty/retraction 改变 action，
  不是 prompt 长度或 target tracker 差异。

## 7. 三条未来方向的独立评估

相关工作边界采用原始来源：MEMI 官方代码/项目页；Ledger
([arXiv:2609.34554](https://arxiv.org/abs/2609.34554))；SimpleARM
([arXiv:2609.36595](https://arxiv.org/abs/2609.36595))；Statler
([arXiv:2306.17840](https://arxiv.org/abs/2306.17840))；probabilistic anchoring
([arXiv:2002.10373](https://arxiv.org/abs/2002.10373))。这些工作分别已覆盖 task-specific learned
memory/policy、外部 object ledger、typed relation/reference/progress/route state、显式 LLM world-state，及
噪声下的 probabilistic symbolic anchoring。ArbiterKB 不能只靠“显式 memory/scene graph”声称 novelty。

### 排名 1 — B：测得感知错误下的概率 belief、retraction 与 calibration（条件 GO）

- **潜在 novelty**：不是“有 probability”，而是 calibrated uncertainty/retraction 是否在闭环中触发
  look/verify/abstain，减少 confident wrong actions；RoboMME 现有 memory 方法很少直接测这个因果链。
- **现有证据**：held-out perception audit 中 full KB 未胜 soft repair；R6 在 KB 内改善 Brier，但没有改善
  top-1，且本次 clean validation 只有 4 个 belief ground-truth rows。信号存在但很弱。
- **最强 baselines**：deterministic relink/cover ledger、SoftHaircut、calibrated tracker probabilities、
  SimpleARM-style typed state、MEMI with identical noisy observations；还应比较 cost-matched active re-observation。
- **最大风险**：概率只是重排同一错误 evidence；降低 confidence 但不改变 action；R5/R1 把相关弱 evidence
  重复计算；效果仅出现在人工 corruption。
- **GO**：在预先测量的自然 tracker errors 上，calibration 与 closed-loop success 同时改善，且胜过 soft
  baseline；action mediation 显示增益来自 verify/replan。
- **NO-GO**：soft repair 在 AURC/Brier/success 上相当；belief confidence 不改变动作；效果只在设计的
  synthetic corruption 出现。

### 排名 2 — C：通用 memory 的 cross-task / compositional generalization（高门槛、条件 GO）

- **潜在 novelty**：真正未见任务/未见组合上，统一 memory ontology 是否比 per-task optimized MEMI prompt
  更少 adaptation、更稳健；重点必须是 held-out task families 或 novel compositions，而不是同四任务。
- **拥挤程度**：Ledger 已主张跨 memory type 的外部 object ledger；SimpleARM 已用 instruction-derived typed
  state；Statler 已维护通用 world state。单纯“generic KB”不新。
- **最强 baselines**：MEMI zero-shot/retuned、Ledger、SimpleARM、Statler-style writer/reader、neutral
  deterministic ledger、FullHist，以及 direct program/FSM。
- **最大风险**：通用 ontology 缺少 task-progress semantics；per-task prompt 用少量 demos 即可补齐；KB 的
  object richness与任务 success 无关，反而增加 token 与 planner confusion。
- **GO**：在严格 held-out task/composition 上，零任务调参的通用 memory 以明确 margin 胜过 retuning-budget
  matched baselines，并保持成本/错误可解释性。
- **NO-GO**：每个新 task 仍需新增 rule/counter/prompt；简单 instruction parser + typed ledger 相当；优势来自
  训练/提示预算不公平。

### 排名 3 — A：修复 ArbiterKB task-progress 与 policy（工程 GO，论文 NO-GO）

- **价值**：接上 execution-only counter、分离 demo/execution、加入明确 termination rule，很可能修复
  VideoRepick；也能让 harness 更公平。
- **novelty 很低**：MEMI prompt、SimpleARM progress state、Ledger 的 elapsed cycles、直接 FSM 都已覆盖。
- **最强 baselines**：direct FSM、deterministic action ledger、M0P1、FullHist P1。
- **最大风险**：每个 task 加一个 parser/FSM，最终是 benchmark-specific policy library，而不是通用 memory；
  success 上升却没有任何 belief/KB 主效应。
- **GO（仅作为实验控制）**：实现 factorial 所需的 matched policy，冻结后用于消除 confound。
- **NO-GO（研究方向）**：若 deterministic FSM/ledger 达到同等结果，停止把它作为 ArbiterKB 贡献。

## 8. 对 ArbiterKB 架构的判决

当前证据允许、也要求一个不讨好的结论：**现有 full ArbiterKB 可能是过度设计。**

- clean closed loop：不胜 FullHist，0/9 VideoRepick；
- held-out perceived belief：不胜 detector-only soft repair 的 accuracy/AURC/Brier；
- rule ablation：部分规则无覆盖或可能伤害 decision；
- 当前最清楚的失败只需 execution-only counter/FSM，完整 belief graph 不解决它；
- 当前最清楚的 identity signal（VideoRepick 9/9 accepted target）没有转化为 task success。

这不证明所有 probabilistic KB 都无用；它证明在继续消耗 held-out、扩规则或包装论文 claim 前，当前架构
必须先击败 simple ledger/FSM falsifier。否则应保留其中可复用的 tracker、event schema 和 calibration
工具，放弃“完整五层 KB 是必要 memory architecture”的主张。

## 9. 单一最有价值的下一实验

**运行第 6 节冻结的 2×2 factorial，并把 deterministic demo-to-FSM/cover ledger 设为硬 baseline；先只用
非 held-out validation/development episodes，不做 degradation。**

核心假设是：`ArbiterKB structured state` 在 task policy 匹配后仍提供超过 deterministic neutral ledger 的
独立、可重复增益。最直接的 falsifier 是：`M0P1 ≥ M1P1`，KB 主效应 CI 不包含预注册的实用增益，且
direct FSM 在 VideoRepick 达到同等或更高 success。若出现这个结果，就不应再修完整架构；研究资源应转向
更窄的“测得 perception error 下，calibrated uncertainty 是否改变 action”问题，或停止该分支。

之所以先做这个实验，而不是立刻做 noisy perception，是因为当前最大不确定性不是 simulator 能否运行，
而是**我们到底在比较 memory 还是 task policy**。不先消除该混杂，任何 degradation 结果都无法告诉我们
KB 的增量价值。

## 10. 重要 claim → 证据映射

| claim | 直接证据 |
|---|---|
| 144/144 终态、seed/reset/model/integrity PASS | `reproducibility_manifest.json`；`summary.json#audit` |
| 总体 success、paired tests、Holm family | `summary.json#summary` 与 `#paired_tests`；本报告由 raw outcomes 重算于 `failure_forensics.json#paired_tests` |
| 36 条 VideoRepick choice/point/status/divergence | `failure_forensics.json#video_repick`；原始 `validation/clean/VideoRepick_ep{1..9}_{arm}/result.json` |
| Arbiter 9/9 首抓 accepted，35/35 decisions executed | 上述九个 Arbiter `result.json#decisions`；`failure_forensics.json#video_repick` |
| demonstration count 已在 decision 0 | 九个 Arbiter `events.jsonl` 的 `phase_start`, `phase=llm`, `decision=0`, `memory_before.text` |
| ep1/3/5 提前完成的显式 reasoning | 对应 Arbiter `result.json#policy.log[-1].reasoning` |
| ep6/9 错误要求再抓的显式 reasoning | 对应 Arbiter `result.json#policy.log[-1].reasoning` |
| common harness 未注入 goal counter | `arbiter_kb/common_memory.py:40-43`；`arbiter_kb/systems/arbiter_kb.py:30-35`；`arbiter_kb/views.py:192-208` |
| MEMI 与 Arbiter prompt 不等价 | `stress/common_policy.py:21-50,92-104`；pinned MEMI `memi/plan/rollout/policies.py:121-183`；`optimized_luna.json[VideoRepick]` |
| 全部 completed decisions executed；无 executor/API/harness error | `failure_forensics.json#behavior`；每个 raw `result.json#decisions`; `reproducibility_manifest.json#integrity` |
| loops 只在 ButtonUnmaskSwap | `failure_forensics.json#loops`；对应 `events.jsonl` 的 `loop_detected` |
| belief scorer 只有 VUS 4 rows/arm | `summary.json#beliefs` |
| full KB 未胜 soft perception repair | `paper/PERCEPTION_BELIEF_COMPARATIVE_REPORT.md`；`paper/PAIRED_ERROR_ABLATION_AUDIT.md` |
| 原始数据未改，派生分析可追溯 | `failure_forensics.json#source_sha256`；script 只读 raw tree 后写独立 output |

### 复算命令

```bash
python3 scripts/robomme_failure_forensics.py \
  results/robomme_common/watchdog_v2_1/validation_20261031T150046Z/validation/clean \
  --out results/robomme_common/watchdog_v2_1/validation_20261031T150046Z/failure_forensics.json
```

该命令不访问网络、模型、模拟器或 held-out data。
