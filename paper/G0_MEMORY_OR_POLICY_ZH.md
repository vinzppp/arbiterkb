# G0：「Memory or policy?」主研究的免费闸门

日期：2026-11-01
分支：`research/perception-belief-robomme`（本报告提交见 git log；`main` 仍为 `069beb7`）
依据：主仓库 NOTES「Memory or policy? Preregistration of the main study」（7327cf1）的 G0；
docs/HANDOFF_ORCHESTRATOR.md「Paper direction」（2fd5bda）；docs/GOALS.md（88287a5）
费用：$0。没有调用 LLM，没有读 test split 的内容，没有跑 test episode，没有连 pod。
机器可读结果：`results/memory_or_policy/g0/summary.json`

## 结论先行

| G0 项 | 判定 | 一句话 |
|---|---|---|
| 1 episode 供应 | **通过（VideoUnmaskSwap 的 G1 需要你决定）** | train 只有 0–99，全在 MEMI 训练数据里，train 不用；val 剩 RS/SX/SC 各 49、VR 40、VUS 严格 0 / 宽松 20；test 0–49 五个任务都可用 |
| 2 五个任务能否跑通 | **通过，带两处输入缺口** | 5/5 走通采集、执行器、FullHist / MEMI / ledger / KB 四条 prompt 路径；但 bridge 丢掉了 button、waypoint、highlighted target/area，9 臂里有 5 个尚未实现 |
| 3 MEMI prompt 拆分 | **脚本部分通过；人工部分待做** | 规则 1–2 脚本拆分两种实现逐字节一致、重复运行一致；规则 5 的人工拆分与规则 4 的逐句标注文件已备好，等人来做 |
| 4 KB 进度 bug 修复 | **通过（补了一处漏洞）** | 修复和回归测试在分支上；G0 发现 hold 事件行仍带 demo 计数（“left hold #2”），已补并加测试 |
| 5 计算时间 | **Mac 实测完成；pod 待你启动后实测** | Mac 单 episode 67–178 s、约 1.5 核，3 路并行只快 1.07 倍；G1 约 13–27 h，正式验证（n=250）约 134–267 h 串行，需上 pod |
| 6 G2（描述性） | **完成：0 / 35 决策换目标** | stock 误差经执行器吸附后 0% 换目标（全帧 0 / 4,276）；低于 2% 阈值，按方向文件感知轴放弃，作为发现报告 |

**有一项会影响结论的解释，需要 orchestrator 看：** MEMI 的 subgoal 文本（即 P-matched）在 5 个任务里都直接引用
MEMI 自己的记忆字段（如 RouteStick「First update the remaining path according to the policy prompt for memory. If
its first letter is A, select A」）。字段名替换后，这些句子对 FullHist、ledger、KB 三个记忆臂指向不存在的东西。
所以 P-matched 不是与记忆无关的「纯策略」，H1/H2 的对比里策略和 MEMI 的记忆结构是耦合的（§3.4）。这不改变任何
规则，但需要在 G1 之前决定怎么说明。

## 1. Episode 供应

脚本：`scripts/g0_episode_supply.py` → `results/memory_or_policy/g0/episode_supply.json`。
输入只有元数据：MEMI `data/decisions/<Task>.jsonl`（bb6c37c，从 Docker 镜像 `arbiterkb-robomme-cpu:bb6c37c` 的
`/opt/memi` 复制，`git rev-parse` = bb6c37ce…）、RoboMME `env_metadata/{train,val,test}`（fae9f73；字段只有
episode、seed、difficulty）、本分支的 result 行、主仓库的使用记录。

### 1.1 MEMI 用过的 episode（16 个文件全查）

16 个 `data/decisions/<Task>.jsonl` 每个都是 100 个 episode，编号 0–99，连续、无重复，MEMI 自己按哈希切成 90 train
/ 10 dev。来源是 RoboMME 的录制数据 `record_dataset_<Task>.h5`（`memi/plan/data/build_decisions.py`；MEMI README
指向 HuggingFace `Yinpei/robomme_data_h5`）。RoboMME 的 train 元数据每个任务恰好 100 个 episode（0–99），
**没有 0–99 以外的 train episode**。因此按预注册，**train split 完全不用**。

同一任务内 train / val / test 的 seed 两两无重叠（16 个任务都查了）。

### 1.2 本项目和本分支已经跑过或读过的

| 来源 | episode |
|---|---|
| 本分支 v1 / v2.1 / factorial（result 行扫描） | VideoUnmaskSwap、ButtonUnmaskSwap、PickHighlight、VideoRepick 的 val 0–9 |
| 主仓库 Astra pilot（NOTES） | VideoUnmaskSwap val 0–9 |
| 主仓库 two-pick K run（POD_STATE.md） | VideoUnmaskSwap val 11 15 16 17 19 23 24 25 27 29 31 32 35 37 39 40 41 43 44 45 |
| 主仓库 goals 列表（`cl/goals.jsonl`） | VideoUnmaskSwap val 10–49 **只读了目标文本**（reset 一次，选 two-pick；没跑策略、没看结果） |
| 主仓库 phrase check | VideoPlaceOrder val 0–4（不是本研究的任务） |
| 本分支 LeRobot 计数实验 | SwingXtimes、VideoRepick 等的 **train**（公开 LeRobot 数据，train 0:1600），反正已排除 |
| 本次 G0 preflight | RouteStick、SwingXtimes、StopCube 的 val 49（取最高编号，留出最低编号给 G1） |

另外：本分支早先解析过 MEMI 公开发布的 **test** 日志（`results/latest/paper_comparison`，MEMI 与 FullHist 的逐
episode 结果）。预注册已经把「任务按 MEMI 公布的 test 数字选」写成已知偏差；我们自己的任何臂都没有在 test 上跑过，
`results/test_reads.log` 里没有 RoboMME test 的 READ。

### 1.3 每个任务能用的 episode

| 任务 | val 总数 | 已用 | val 剩余（严格） | val 剩余（只读目标不算用过） | G1 用的 5 个（最低编号） | test 0–49 |
|---|---:|---|---:|---:|---|---|
| RouteStick | 50 | 49（G0） | 49 | 49 | 0–4 | 可用：50 个 seed 齐全，从未跑过 |
| SwingXtimes | 50 | 49（G0） | 49 | 49 | 0–4 | 可用 |
| VideoUnmaskSwap | 50 | 0–9、20 个 two-pick 跑过；10–49 读过目标 | **0** | 20 | 严格：无；宽松：10 12 13 14 18 | 可用 |
| StopCube | 50 | 49（G0） | 49 | 49 | 0–4 | 可用 |
| VideoRepick | 50 | 0–9 | 40 | 40 | 10–14 | 可用 |

test 的难度分布 5 个任务都是 easy 26 / medium 12 / hard 12。test 能否 reset：RoboMME 的 `BenchmarkEnvBuilder` 只从
这些元数据取 seed 和 difficulty（`episode_config_resolver.py`），50 个记录全部有 seed，所以可以确定性 reset；我没有
reset 任何 test episode。注意 RoboMME 对超出 split 的编号会静默生成默认场景（主仓库 POD_STATE.md），现有
harness 的 `episode_seed` 会对此报错，正式 manifest 应保留这道检查。

**判定：通过。** 正式验证用 test 0–49 × 5 任务 = 250 对，供应足够（n = min(250, 供应) = 250）。
**需要你决定：** VideoUnmaskSwap 没有严格意义上未用过的 val。两个选项：(a) 把「只读过目标文本」的 episode 算作
未用，G1 用 10 12 13 14 18——它们没跑过任何策略、没看过结果，但都是被筛掉的单目标 goal，比 two-pick 容易，G1
的 discordance 估计会偏低；(b) VideoUnmaskSwap 在 G1 里不跑，只用其余 4 个任务的 d 定 n。我倾向 (a)，并在 G1
报告里注明这层选择。预注册的任务替换规则只覆盖「harness 跑不了」，不适用于这里。

## 2. 五个任务在 harness 里能否跑通

脚本：`stress/g0_preflight.py`，在 `arbiterkb-robomme-cpu:bb6c37c` 里以 `--network none` 运行，环境里没有任何 key
（脚本发现 key 会拒绝运行）。执行器由脚本化选择驱动：第 i 次决策选菜单第 i mod |menu| 项，需要点时用该项第一个
可用点或分段里最后一个像素，最多 6 次决策。每次决策时，同一分段还送进 FullHist、stock MEMI、ledger、KB 四条
prompt 路径，用一个桩（stub）客户端返回同一个脚本答案，所以每条路径都构造了真实 prompt，但没有网络调用。
输出：`results/memory_or_policy/g0/preflight/<Task>_val<ep>/`（每臂第一和最后一个 prompt 全文、ledger/KB 最终记忆、
逐项计时）。

| 任务（episode） | 决策 | 执行结果 | 结局 | 采集到的物体类型（MEMI 分类） | ledger/KB bridge 保留的 |
|---|---:|---|---|---|---|
| VideoUnmaskSwap（val 0） | 1 | executed | fail（脚本选错容器） | 3 色 cube、container | cube、container |
| VideoRepick（val 0） | 1 | executed | fail | container（实为 cube）、button | cube（bin→cube 的任务特定改名） |
| RouteStick（val 49） | 1 | executed | fail | cube（桌上的 stick）、target、waypoint、highlighted target | cube、target |
| SwingXtimes（val 49） | 4 | 4 × executed | fail | red cube、target、button、highlighted area | cube、target |
| StopCube（val 49） | 3 | 3 × executed | fail | cube、target、button | cube、target |

所有 10 条执行器命令都返回 executed，没有 api_error、planner_error、harness_error，没有采集错误。四条 prompt
路径在 5 个任务的每次决策上都构造成功；P-generic 的 system 文本在 ledger 与 KB 之间逐字节相同
（system sha256 `e182003e…`）。「fail」是脚本选择的结果，不是 harness 故障。

**判定：通过（采集、执行器、四条 prompt 路径）**，但有下面这些缺口，必须在 G1 之前处理：

1. **bridge 丢类型。** `stress/common_scene.py::_label` 只保留 cube / container / target。button（VR、SX、SC）、
   waypoint 与 highlighted target（RS）、highlighted area（SX）进不了 ledger 和 KB。按钮在三个任务里不需要点
   （菜单项无 `available`），但 RouteStick 的路线信息（demo 中手经过哪些 waypoint、绕向）完全不在 ledger/KB 里，
   这两臂在 RS 上几乎是空记忆。另有 VideoRepick 的 bin→cube 改名是任务特定代码。建议：bridge 改成保留 MEMI
   capture 命名的所有类型（任务无关），不替换任务。是否替换 RouteStick 按预注册是 G0 文档化的判断，我把它列为
   待 orchestrator 决定，倾向不替换。
2. **9 个臂里只有 4 个已有实现。** 已有：FH-g 所需的 FullHist（MEMI 原版）、stock MEMI、P-generic 文本、以及
   factorial 的 ledger/KB 记忆。缺：(a) 预注册规格的 M-ledger（按选项记录的执行日志与结果、按选项类型计数、单独
   标「demonstration」的 demo 块）——现有 `DeterministicWorldLedger` + `NormalizedTaskState` 记的是检测器事件，
   且带一个 VideoRepick 风格的「Demonstration-grasped cube」推断，不是预注册的样子；(b) M-MEMI（MEMI fields +
   任意策略）；(c) 从拆分结果生成的 P-matched（现有 matched 文本是前任手写的，只覆盖 VR 和 VUS）；(d) 把
   P 文本注入 FullHist 的 FH-g / FH-m。
3. **超时规则要换新 manifest。** v2.1 的 episode 上限是 600 s，预注册是 30 min 后备 + 单次执行器 300 s + 50 次
   调用（MEMI `MAX_DECISIONS = 50`，已一致）。

## 3. MEMI prompt 拆分

脚本：`scripts/memi_prompt_split.py`，输入 MEMI bb6c37c 的 `artifacts/gpt/policy_prompts/optimized_luna.json`
（sha256 `307eaf7f…6fda54`）。输出 `results/memory_or_policy/g0/memi_split/<Task>/{policy_matched.txt,
memory_fields.json}` 与 `split_summary.json`。

### 3.1 规则 1–2 的脚本拆分

记忆 = `fields[]` 原样；策略 = 整段 `policy_prompt_for_subgoal`，不移动任何句子；策略里每个 `fields[].name`
标识符（按整词匹配，下划线算词内字符）替换成「the 」+ 名字（下划线换空格）。

| 任务 | 字段 | 替换次数 | policy sha256（前 16 位） | fields sha256（前 16 位） |
|---|---|---|---|---|
| RouteStick | remaining_path | 2 | b12e3f0bbbc6f304 | 0d807b5d9880339c |
| SwingXtimes | completed_target_visits | 3 | b40218d1e18cb2b4 | 8c2a2d1a21ae9393 |
| VideoUnmaskSwap | remaining_targets | 2 | 6cb71d9e59a3a4de | 153fcd5121b31a86 |
| StopCube | passage_count / last_side / motion_history | 5 / 0 / 1 | db0724eddb6cd9f5 | 2824b527396ab1a8 |
| VideoRepick | target_anchor / completed_cycles | 3 / 5 | b784d92a15e35217 | 5ebf8e3c3cf24d93 |

### 3.2 一致性

- 两种独立实现（正则替换、逐字符扫描标识符）在 5 个任务上**逐字节一致**，替换后策略里不再出现任何字段名。
- 脚本重复运行，输出目录逐字节一致（`diff -r`）。
- 这只是脚本的自检。**预注册规则 5 要求的第二次拆分是「一个人不看脚本输出、手工按规则 1–2 拆」**，还没做。
  工作表：`results/memory_or_policy/g0/labeling/memi_hand_split_worksheet.md`（5 个任务的原文与规则，留空填写）。
  填好后与 `policy_matched.txt` 比较（空白规范化后逐字节），不一致就是 bug，修好再查。

### 3.3 规则 4 的逐句标注（由你做）

`results/memory_or_policy/g0/labeling/memi_subgoal_sentence_labels.csv`：98 句（RS 12、SX 16、VUS 10、SC 37、
VR 23），列 `coder_1`、`coder_2`、`adjudicated`、`note` 留空。类别按预注册：*choice/termination/progress* 与
*perception/update*。断句只按「句号/问号/感叹号 + 空格 + 大写或引号」机械切分，标注者觉得断错可在 note 里写。
两位标注者应在看到任何结果之前独立标注。

### 3.4 机械规则带出来的两点（描述性，不改规则）

1. **P-matched 引用 MEMI 的记忆。** 5 个任务的 subgoal 文本都按名字使用 MEMI 字段，且要求先更新记忆：
   RouteStick「First update the remaining path according to the policy prompt for memory. If its first letter is
   A, select A …」；VideoUnmaskSwap「Apply the memory update before choosing the subgoal」；VideoRepick「the
   completed_cycles policy prompt」等。替换后，对 FH、LD、KB 三臂这些短语指向它们 prompt 里不存在的字段说明。
   后果：MM-m 与其他 ·-m 臂的差别不只是「记忆结构」，还包括「策略能否找到它引用的记忆」。这正好是 H2 想检验的
   东西被混进了策略侧；也让 H1 的「策略」在 RouteStick 上几乎等于「按 MEMI 记忆里的字母序列走」。建议
   orchestrator 在 G1 前写一段带日期的说明：要么接受并在每张表注明，要么把规则 4 的标注扩成三类（加
   memory-reference），并报告每个任务这类句子的比例。
2. **一处「the the」。** VideoRepick 原文「in the completed_cycles policy prompt」替换后成「in the the completed
   cycles policy prompt」。规则 2 的字面结果如此；人工拆分若顺手去掉一个 the，就会和脚本不一致——两边都应按字面
   规则保留。StopCube 的 `last_side` 在 subgoal 里没出现（0 次替换）。

**判定：脚本拆分通过；规则 5（人工拆分）与规则 4（双人标注）未完成，属于人工项。**

## 4. KB 进度 bug 的修复

### 4.1 修复在分支上的位置

预注册要求 KB 臂使用 `phase_safe_arbiter_view`（去掉含 demo 的进度）。分支上已有：

- `arbiter_kb/factorial_memory.py::phase_safe_arbiter_view`（fd62600）：从 KB 的 `controller_view` 文本里删掉以
  `Progress:` / `Goal progress:` 开头的行，以及含操作转录短语（` event grasp(`、` picked up an object ` 等）的行；
  `FactorialMemory.text()` 对 KB 臂套用它。
- `NormalizedTaskState`（091be07，factorial 预注册时加入）：demo 与 execution 各用一个 `Detectors`，只把
  execution 的 pick_up / release / press / hold_above 计入「Execution progress」，ledger 与 KB 共用同一段文本。
- 回归测试（`tests/test_factorial.py`）：`test_normalized_task_state_excludes_demo_from_execution_progress`、
  `test_arbiter_adapter_suppresses_demo_progress_from_legacy_view`。

### 4.2 G0 发现的漏洞并已修

`action_view.action_rows` 把 hold 事件按 demo + execution 一起编号，后面没有紧跟放下时打印成独立一行
「held the object up on the robot's left side (left hold #2)」。过滤器只认合并行的「 held it up」，独立行漏了。
用一个最小 fixture 复现：demo 一次「抓起 → 左侧举起停住」，execution 再一次；修复前过滤后的视图同时出现
「Execution progress: … holds 1」和「(left hold #2)」。VideoRepick val 0 的 demo 里恰有一次 left hold
（原始视图「Progress: grasps 1, places 1, pick-ups 1, releases 1, holds left 1, right 0」），所以 VR 的 KB 臂在
execution 中举着方块做决策时会看到 demo 计入的序号。

修复：`phase_safe_arbiter_view` 的过滤短语加上 `" held the object up on the robot's "`；新测试
`test_arbiter_adapter_hides_demo_numbered_hold_lines`（修复前失败、修复后通过）。

### 4.3 对 5 个任务是否生效

| 任务 | 有 demo | 原始 KB 视图里的 demo 计数（G0 preflight） | 过滤后 |
|---|---|---|---|
| VideoUnmaskSwap | 是 | `Progress: covers 3` | 删除；cover 关系行保留 |
| VideoRepick | 是 | `Progress: grasps 1, places 1, pick-ups 1, releases 1, holds left 1` | 删除；Execution progress 为 0 |
| RouteStick | 是 | `Progress: nothing yet`（demo 中无抓取，只有 target 被遮挡事件） | 删除 |
| SwingXtimes | 否 | 只有 execution 计数 | 无 demo，不存在泄漏路径 |
| StopCube | 否 | `Progress: nothing yet` | 同上 |

SwingXtimes 和 StopCube 没有 demo（MEMI `has_demo = false`，preflight `demo_rows = 1`），这个 bug 在它们上不会发生。
修复后代码的 5 任务 preflight 见 `results/memory_or_policy/g0/preflight_parallel3/`。

**判定：通过。** 精确的修复说明见附录 A。

## 5. 计算时间

### 5.1 Mac 实测（model-free，每任务 1 个 episode）

Mac：Apple M2，8 核，8 GB；Docker 分到 8 CPU、3.8 GiB；镜像是 linux/amd64，在 arm64 上走 x86 转译和 CPU 渲染。

| 任务 | 决策 | 进程启动 | reset（含 demo 与记忆初始化） | 每次执行器 | episode 墙钟 | CPU 时间 | 平均占用核数 |
|---|---:|---:|---:|---|---:|---:|---:|
| VideoUnmaskSwap | 1 | 13.1 s | 41.2 s | 32.7 s | 87.2 s | 137.7 s | 1.58 |
| VideoRepick | 1 | 15.1 s | 105.8 s | 32.4 s | 153.7 s | 234.3 s | 1.52 |
| RouteStick | 1 | 12.4 s | 147.1 s | 18.6 s | 178.4 s | 266.8 s | 1.50 |
| SwingXtimes | 4 | 11.5 s | 0.5 s | 42.2 / 11.6 / 9.2 / 31.3 s | 106.5 s | 168.2 s | 1.58 |
| StopCube | 3 | 10.9 s | 0.5 s | 14.2 / 14.2 / 26.5 s | 66.6 s | 108.6 s | 1.63 |

（出处：`preflight/<Task>_val<ep>/g0_preflight.json`。这一轮 VUS、VR 在修复前、RS/SX/SC 在修复前启动；修复只
改 KB 文本，不影响计时。）

**Mac 并行实测（修复后代码，同样 5 个 episode，3 个容器同时跑）：**

| 任务 | 串行墙钟 | 3 路并行墙钟 | 变慢 |
|---|---:|---:|---:|
| VideoUnmaskSwap | 87.2 s | 132.8 s | 1.52× |
| VideoRepick | 153.7 s | 281.5 s | 1.83× |
| RouteStick | 178.4 s | 457.4 s | 2.56× |
| SwingXtimes | 106.5 s | 356.0 s | 3.34× |
| StopCube | 66.6 s | 245.4 s | 3.68× |

5 个 episode 串行合计 592 s，3 路并行总墙钟 552 s：**吞吐只提高 1.07 倍**。每个容器的 CPU 时间也涨了（例如
RouteStick 267 → 566 s），说明瓶颈是 x86 转译和 3.8 GiB 的 Docker 内存，而不是核数。**Mac 的有效并行宽度按 1
算。** 并行跑出的命令序列、执行结果和结局与串行完全相同（确定性复现）；修复后 KB 文本里没有「hold #」，也没有
`Progress:` 行（`preflight_parallel3/`）。

### 5.2 时数估算

模型（`scripts/g0_summary.py`）：每格 = 进程启动 + reset（上表）+ 调用数 × 每次调用秒数。每次调用 30.5 s =
v2.1 验证（同一镜像、同一 Luna endpoint）的（执行器 108.75 s + LLM 13.34 s）× 144 格 ÷ 577 次调用。调用数用
MEMI 专家示范的每 episode 决策步数（RS 3.69、SX 7.46、VUS 2.08、SC 4.93、VR 5.14；只读 decisions 的步数）。
悲观行：调用数加倍，且 5% 的格跑到 30 min 后备。时数 = 9 臂 × 每任务 episode 数 × 平均每格秒数 ÷ 并行宽度
（假设线性扩展；Mac 实测宽度约 1，宽度 > 1 的列只适用于 pod，需 §5.3 实测确认）。

| | 平均每格 | G1（225 格）串行 | G1 宽度 3 / 6 / 8 | 正式（9 × 250 = 2,250 格）串行 | 正式 宽度 3 / 6 / 8 |
|---|---:|---:|---|---:|---|
| 期望 | 214 s | 13.4 h | 4.5 / 2.2 / 1.7 h | 133.5 h | 44.5 / 22.3 / 16.7 h |
| 悲观 | 428 s | 26.7 h | 8.9 / 4.5 / 3.3 h | 267.4 h | 89.1 / 44.6 / 33.4 h |

核对：同一模型对 VUS 给 118 s/格、对 VR 给 278 s/格，而 v2.1 实测是 VUS 78–100 s、VR 150–194 s（按臂平均），
所以「期望」行偏保守。

### 5.3 RunPod A40 上要实测的（pod 停着，由你启动；我没有连）

目的：pod 是原生 x86（不用转译）、有 GPU 渲染，单 episode 应比 Mac 快，但快多少、能并行几路都没测过。

1. **环境**（照 `stress/Dockerfile.robomme-cpu` 在 pod 上装一个 venv；RunPod 里一般不能跑 Docker）：
   ```bash
   git clone --filter=blob:none https://github.com/memi-robot/memi.git /workspace/memi && git -C /workspace/memi checkout bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0 && git -C /workspace/memi submodule update --init third_party/robomme_benchmark
   git clone --filter=blob:none https://github.com/YinpeiDai/ManiSkill.git /workspace/ManiSkill && git -C /workspace/ManiSkill checkout 07be6fbc66350ddca200abfb0a11b692f078f7fd
   python3.11 -m venv /workspace/mop && . /workspace/mop/bin/activate && pip install torch==2.9.1 torchvision==0.24.1 && pip install -e /workspace/ManiSkill && pip install --no-deps -e /workspace/memi/third_party/robomme_benchmark && pip install -e /workspace/memi opencv-python==4.11.0.86 setuptools==80.9.0
   ```
   核对：`git -C /workspace/memi/third_party/robomme_benchmark rev-parse HEAD` = fae9f73…；`nvidia-smi`；Vulkan
   （主仓库 POD_RUN.md §0F 的修法）；`SAPIEN_RENDER_DEVICE` 不设（用 GPU），另跑一次设为 `cpu` 作对照。
2. **代码**：把本分支打包上传到 `/workspace/ArbiterKB`（不上传 `.env`，pod 上不设任何 key）。
3. **单 episode 与并行宽度**（model-free，用 G0 已用过的 5 个 val episode，不读新 episode）：
   ```bash
   cd /workspace/ArbiterKB && MEMI=/workspace/memi bash stress/g0_pod_parallel.sh "1 2 4 6 8" results/memory_or_policy/g0/pod
   ```
   每个宽度 k 同时跑 k 份 × 5 个任务，记录每 episode 墙钟、`/proc/loadavg`、GPU 利用率与显存，写
   `widths.jsonl`（episodes/hour）。取吞吐不再上升、且单 episode 墙钟不超过 k=1 的 1.5 倍的最大 k。
   `machine.txt` 记 `nproc`、内存、磁盘、GPU 型号。
4. **磁盘**：venv + 三个仓库约 8–10 GB（CUDA 版 torch 约 5 GB；CPU 镜像内容 1.57 GB 作下限）；每格结果约 36 KB
   （v2.1 144 格共 5.2 MB），2,250 格约 80 MB；不录视频。网络卷留 **20 GB** 足够。
5. 跑完把 `results/memory_or_policy/g0/pod/` 拷回 Mac，停 pod。

**判定：Mac 部分完成；pod 部分待实测。** Mac 实际只能串行：G1 约 13–27 h，正式验证约 134 h（5.6 天）到
267 h（11 天）。一个月内勉强可行但没有余量（任何重跑都会挤掉分析时间），所以**正式验证应放在 pod 上**；
pod 宽度 6 时正式验证约 22–45 h。G1 可以在 Mac 上跑（约半天到一天）。

## 6. G2（描述性）：stock 误差经执行器吸附

脚本：`scripts/g2_snapping.py` → `results/memory_or_policy/g0/g2_snapping.json`。只读 G **train**
（`data/g_perceived/<setting>_ep000–009.npz`），后缀 ≥ 10（G test）在打开文件之前就拒绝。

方法：RoboMME `multi_choice` 包装器把点交给 `select_target_with_pixel`，选投影中心离点最近的可选物体
（MEMI `memi/plan/data/candidates.py` 调用的同一函数）。每个感知槽在它首次出现的帧绑定一个 oracle 容器
（`sg_compat.assign_slots_seeded`，即记忆所持有的身份）。一次决策 = 回放最后一帧（所有交换之后）的一个容器：
记忆指向该槽最近一次感知到的像素，吸附到最近的 oracle 容器；吸附结果不是它自己的容器就算「换目标」。全帧作为
敏感性分析。近似：用 oracle 掩码质心代替包装器用的投影中心。

| 设置 | episode | 决策（末帧） | 换目标 | 比例 [Wilson 95%] | 感知像素误差 中位 / p95 / 最大 | 吸附余量 最小 / p5 / 中位 |
|---|---:|---:|---:|---|---|---|
| stock | 10 | 35 | 0 | 0.0% [0, 9.9%] | 4.2 / 11.3 / 13.7 px | 16.7 / 25.1 / 37.3 px |
| stock 相机平移（cam_left/right） | 20 | 70 | 0 | 0.0% [0, 5.2%] | 2.6 / 10.4 / 15.0 px | 18.9 / 23.7 / 38.7 px |
| stock，全帧 | 10 | 4,276 | 0 | 0.0% [0, 0.09%] | 3.7 / 10.7 / 14.7 px | 0.4 / 17.9 / 33.4 px |
| 相机平移，全帧 | 20 | 8,199 | 0 | 0.0% [0, 0.05%] | 2.4 / 11.7 / 15.0 px | 1.2 / 19.0 / 33.0 px |

读法：stock 下感知误差最大约 15 px，而末帧容器之间的吸附余量最小 16.7 px，所以没有一次决策换目标；这和主仓库
NOTES「G — gate result」里 stock 0 / 44 次交换一致。全帧里余量会降到 0.4 px（交换途中两个容器靠近），但那时不是
决策时刻。末帧 n = 35 的 Wilson 上界 9.9% 不能单独排除 2%；全帧（相关样本）上界 0.09%。按方向文件的规则
（少于 2% 换目标则放弃感知轴并作为发现报告），**点估计 0% → 放弃感知轴**。这只覆盖 VideoUnmaskSwap 类的
容器场景（G 只有这一种任务）。

**判定：完成。**

## 7. 有没有哪一项让一个月不可行

没有硬阻断。按影响排序：

1. **5 个臂还没写**（M-ledger 按预注册规格、M-MEMI、P-matched 生成、FH 注入 P），加 bridge 保留全部物体类型、
   新 manifest 与 30 min 超时。估计 2–3 天工作量，加上一轮 code-reviewer + goal-reviewer。这是关键路径。
2. **规则 5 的人工拆分和规则 4 的双人标注**：人工项，98 句，估计各 1–2 小时，必须在 G1 前完成。
3. **P-matched 引用 MEMI 记忆（§3.4）**：不是可行性问题，是解释问题；需要 orchestrator 在 G1 前定。
4. **VideoUnmaskSwap 的 G1 episode**：需要你在 (a)/(b) 间选。
5. **算力**：Mac 并行无效（3 路只快 1.07 倍），串行跑正式验证要 6–11 天；需要 pod。pod 并行宽度未测，
   是正式验证日程里最大的未知数。

时间线草案：第 1 周写臂与复审、人工拆分与标注；第 2 周初 G1（约 $1，需 orchestrator 预检与你的 go）、定 n；
第 2–3 周正式验证；第 4 周分析与写作。

## 附录 A：KB 进度 bug 修复说明（供以后合并 main 用）

**问题（main@069beb7）：** `stress/common_episode.py` 在 reset 时把 demo 与首帧一起送进 `policy.observe`；
`arbiter_kb/systems/arbiter_kb.py` 只建一个 `Detectors`，不分 `frame.is_video`；demo 的抓取/放下/按压/举起进入
同一 `kb.evidence`；`arbiter_kb/views.py::_view_v3` 的 `Progress:` 行与 `action_view.action_rows` 的事件行把它们
当作执行进度和序号显示。

**修复（只在研究分支的适配层，不改 KB 内部、默认 controller view 或任何冻结结果）：**

| 提交 | 文件 | 函数 / 类 | 改动 |
|---|---|---|---|
| 091be07 | `arbiter_kb/factorial_memory.py` | `NormalizedTaskState` | demo 与 execution 各一个 `Detectors`；只把 execution 的 pick_up / release / press / hold_above 计入「Execution progress」；ledger 与 KB 共用此段 |
| fd62600 | `arbiter_kb/factorial_memory.py` | `phase_safe_arbiter_view`；`FactorialMemory.text` | 删 `Progress:` / `Goal progress:` 行与操作转录行；KB 臂的世界记忆改走此过滤 |
| 5bc1df4 | `arbiter_kb/factorial_memory.py` | `phase_safe_arbiter_view` | 过滤短语加 `" held the object up on the robot's "`（未合并的 hold 行带 demo + execution 序号） |

**测试（`tests/test_factorial.py`）：**

- `test_normalized_task_state_excludes_demo_from_execution_progress`（091be07）：demo 闭合确定目标方块但不计数；
  execution 的一次稳定抓取计 1。
- `test_arbiter_adapter_suppresses_demo_progress_from_legacy_view`（fd62600）：只喂 demo 抓取；原始
  `controller_view` 有「pick-ups 1」，适配后为「Execution progress: pick-ups 0」，无 `Progress:`、无
  「picked up an object」。
- `test_arbiter_adapter_hides_demo_numbered_hold_lines`（5bc1df4）：demo 与 execution 各一次左侧举起；原始视图有
  「left hold #2」，适配后为「holds 1」，无「hold #」、无「held the object up」。修复前失败。

**合并 main 时注意：** main 上直接用 `controller_view` 的路径（waypoint 线的 K 臂）仍有此 bug；这个修复只在
`FactorialMemory` 适配层生效。若要在 main 的 KB 本身修，应在 `ArbiterKBSystem.step` 按 `frame.is_video` 分开
检测器，并重新冻结 params / prompts（主仓库规则）。

## 附录 B：复现

```bash
# 元数据与 MEMI 文件来自镜像（bb6c37c / fae9f73）
docker run --rm --platform linux/amd64 -v "$OUT:/out" arbiterkb-robomme-cpu:bb6c37c bash -c 'cp -r /opt/memi/data/decisions /opt/memi/artifacts/gpt/policy_prompts /opt/memi/third_party/robomme_benchmark/src/robomme/env_metadata /out/'
.venv/bin/python scripts/g0_episode_supply.py --memi-decisions $OUT/decisions --env-metadata $OUT/env_metadata --g0-preflight '{"RouteStick":[49],"SwingXtimes":[49],"StopCube":[49]}' --out results/memory_or_policy/g0/episode_supply.json
.venv/bin/python scripts/memi_prompt_split.py $OUT/policy_prompts/optimized_luna.json --out results/memory_or_policy/g0/memi_split --labels results/memory_or_policy/g0/labeling
PYTHONPATH=. .venv/bin/python scripts/g2_snapping.py --out results/memory_or_policy/g0/g2_snapping.json
docker run --rm --platform linux/amd64 --network none -v "$PWD:/workspace/ArbiterKB" arbiterkb-robomme-cpu:bb6c37c python stress/g0_preflight.py --task VideoUnmaskSwap --episode 0 --out results/memory_or_policy/g0/preflight
.venv/bin/python scripts/g0_summary.py --expert-calls '{"RouteStick":3.69,"SwingXtimes":7.46,"VideoUnmaskSwap":2.08,"StopCube":4.93,"VideoRepick":5.14}' --status '...'
```
