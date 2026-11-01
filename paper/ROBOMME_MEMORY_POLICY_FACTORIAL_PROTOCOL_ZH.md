# RoboMME Memory × Policy 因子试验预注册协议

状态：**冻结前预注册**  
分支：`research/perception-belief-robomme`  
父协议：RoboMME common-harness v2.1，manifest `a839ccc592f8aa55`  
本协议只用于 development/validation pilot，不涉及 held-out test 或 observation degradation。

## 1. 研究问题

主要问题是：在任务策略、模型、观测、planner、executor、episode 与预算相同的条件下，完整的
ArbiterKB 概率世界状态是否比一个可信但最简的确定性 ledger 带来更高的闭环任务成功率？

2×2 设计如下：

| 条件 | Memory | Policy |
|---|---|---|
| A `ledger_generic` | deterministic ledger | competent generic |
| B `ledger_matched` | deterministic ledger | task-matched |
| C `arbiter_generic` | frozen ArbiterKB | competent generic |
| D `arbiter_matched` | frozen ArbiterKB | task-matched |

主要对比为 D−B。次要对比为 B−A、D−C、C−A，以及交互
`(D−C)−(B−A)`。MEMI 不进入本试验，因为它同时改变 memory 与 task policy，不能作为
memory-only baseline。

## 2. 样本与冻结选择规则

任务固定为 `VideoRepick` 与 `VideoUnmaskSwap`。每个任务使用三个成对 episode，四个条件共享
episode、reset 解析、observation corruption seed 与 executor 设置，共 2×3×4=24 个 scored cells。

机械选择规则是在 RoboMME `val` split 中按 episode index 字典序取前三个：`[0,1,2]`。
此前 common-harness smoke 使用过 index 0，validation 使用过 1–9，因此当前没有未使用的
validation seed。为了不接触 held-out test，本试验明确复用 `[0,1,2]`，并将全部推断限定为
**探索性 pilot**；不能把它称作独立验证，也不能据不显著结果宣称等价。

共同 observation corruption seed 固定为 `73201`，模式为 clean、dropout 0、jitter 0。

## 3. 固定干预

### 3.1 确定性 ledger

Ledger 接收与 ArbiterKB 相同的匿名视觉轨迹，不读 simulator actor identity。它只保存：

- persistent visual-track id、类别、最后位置与可见性；
- cube 消失时与邻近 container 的确定性 cover relation；
- container 轨迹移动时不改变该 relation；cube 再出现时清除 relation；
- 最近事件历史；
- 证据不足时显式 `unknown`；
- 与 ArbiterKB 共用的 execution-only progress schema。

它不使用概率、校准、soft haircut 或复杂 retraction。该设计刻意是一个强而简单的 baseline，
不是为了让 ArbiterKB 获胜而削弱的 strawman。

### 3.2 ArbiterKB

使用现有 `ArbiterKBSystem`、现有参数和冻结规则，不根据 pilot 结果改规则。唯一新增的是共同 policy
adapter、共同 task-state schema 与审计输出。其 controller view 仍由现有 KB 生成。

### 3.3 共同 task state

两个 memory arm 都接收完全相同的 normalized schema：

- demonstration-grasped cube（若观测支持，否则 unknown）；
- execution-only `pick_up`、`release`、`press`、`hold_above` 计数；
- demo detector 与 execution detector 物理分离，演示动作永不计入执行进度；
- 可见 object-to-pixel lookup。

此 schema 是 policy state，不是某一 memory arm 的额外世界知识，并在调用日志中与 memory state
分开记录。两臂都没有 hidden task-progress oracle。

### 3.4 Policy

Generic policy 是任务无关的 competent policy：给出完整 goal、合法 action menu、执行反馈、演示与执行
分离、完成全部停止条件、未知时不得编造 identity。它不包含任务 FSM。

Matched policy 在同一任务的两个 memory arm 中逐字节一致：

- `VideoRepick`：从 goal 读取 N；一次 cycle 是成功 A pickup 后成功 B put-down；demo 不计数；held 时
  选 B；release 后未到 N 选 A，到 N 才选 C；只有执行反馈支持时更新进度。
- `VideoUnmaskSwap`：从 goal 读取目标颜色顺序；利用 memory 输出的 cover relation 与 persistent record；
  非最终目标成功 A 后用 B 放下容器，最终 A 后完成；policy 不提供 container identity 或位置。

Policy SHA-256 已写入 manifest。Memory field 名称差异只存在各自 memory snapshot 内，共同 task-state
字段与 policy bytes 不变。

## 4. 共同运行条件

- wire model：`openai/gpt-6-luna`；不允许 fallback 或静默换模型。
- endpoint：冻结 OpenRouter chat-completions endpoint。
- planner：MEMI 官方 pinned code 的同一 `RolloutClient`、`fullhist` base system 与 `decide_turn`。
- executor：RoboMME `multi_choice`，四臂完全相同。
- effort `low`，max tokens 32768，request timeout 45s，至多 2 attempts。
- 每 cell 新 process group、新 simulator、无 reset retry、无 cell rerun。
- episode 600s、reset 360s、LLM call 120s、executor command 180s、最多 20 个 high-level actions。
- watchdog timeout 是 scored failure；reset/API/worker/host/simulator initialization error 是
  infrastructure error，不作为 task failure，但保留且使完整性 gate 失败。
- API pilot 总 recorded cost 达到或超过 `$0.10` 时停止；此前实验费用不计入。

## 5. 泄漏与公平性 gate

任何 scored query 前必须同时满足：

1. 当前分支正确，`main` 仍为 `069beb7`，源码已提交且 worktree 对实验源码为 clean；
2. manifest 可解析且正好描述 24 个 scored cells；
3. Luna endpoint probe 返回精确模型 ID，无 fallback；
4. pinned MEMI、RoboMME、ManiSkill、params 与 prompt hash 匹配；
5. 两个任务的 scripted simulator/executor preflight 均有合法终止记录；
6. generic pair 与 matched pair 的 policy SHA 相等；四臂 action/token/timeout/episode budgets 相同；
7. 无 held-out/degradation stage；无 simulator actor id、ground-truth cover map 或 task-progress oracle；
8. unit tests、watchdog tests、quality check 与完整测试套件通过。

任一 gate 失败则 NO-GO，不运行 24 scored cells。Credential 只从授权 Keychain item 注入环境，不打印、
写盘或提交。

## 6. 结局与统计方法

主要结局是 simulator task success。`success`、`task_failure`、watchdog `timeout` 都进入 ITT，其中 timeout
计失败。Infrastructure error 与 incomplete 不计 task success，但使 pilot 不完整，禁止选择性重跑。

每臂、每任务报告成功数/3与 Wilson 区间；pooled 仅作描述。四个冻结 pairwise contrasts 使用相同
task-episode block 做：

- paired risk difference；
- 10,000 次 deterministic paired-cluster bootstrap 95% interval；
- two-sided exact paired McNemar；
- 四个 pooled pairwise tests 采用 Holm correction。

Main memory effect 定义为 `mean(C−A, D−B)`；main policy effect 为 `mean(B−A, D−C)`；interaction
为 `(D−C)−(B−A)`，均在六个 task-episode blocks 上成对计算并给 bootstrap interval。任务内 n=3 的
结果只描述，不做强推断。10 percentage points 是未来充分把握 validation 的 practical threshold，
不是本 pilot 的显著性阈值。

同时报告 outcome matrix、failure reason、progress-count error、identity error、action/loop、invalid action、
tokens/cost，以及 reset、planner、executor、total wall time。所有 traces 与事件日志原样保留。

## 7. GO/NO-GO 判据

运行 gate 的 GO 仅表示可以执行 pilot。完成后的 integrity GO 要求：24/24 cells 都有协议认可的 terminal
status、模型 ID 正确、相同 block 的 reset digest/seed 匹配、无系统性 harness error、总费用低于 cap。

研究结论判据：

- 若 D 与 B 的观察结果相当且 ledger 更简单，建议优先 ledger；不能声称统计等价。
- 若 D>B，先审计差异是否来自 uncertainty/retraction/identity 或信息泄漏，未有因果证据前不宣称 KB
  mechanism advantage。
- 若 B、D 都失败，优先检查 policy/executor/representation，而不是扩张 KB rules。

只有 pilot 给出可信、可复现、不是 leakage 的 memory effect 信号，才建议另行冻结更大 validation。

## 8. 冻结与不可变更项

本协议、manifest、memory adapter、policy bytes、episodes、seeds、预算与分析代码在任何 scored call 前提交。
看过 pilot 表现后不得改变 prompt、规则、episode 或预算；不得选择性重跑；不得把旧 v1/v2.1 结果并入
本 pilot。所有负结果、timeout 与 infrastructure error 均写入报告。
