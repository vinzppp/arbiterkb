# G1 包：「Memory or policy?」pilot（待 orchestrator 预检，未运行）

日期：2026-11-01
分支：`research/perception-belief-robomme`（提交见 git log；`main` 仍为 `069beb7`）
依据：主仓库 NOTES「Memory or policy? Preregistration of the main study」（7327cf1）的 G1；
G0 报告 `paper/G0_MEMORY_OR_POLICY_ZH.md`；orchestrator 2026-11-01 的决定（在 Mac 上跑，上限 $2）。

## 1. 跑什么

| 项 | 值 |
|---|---|
| manifest | `experiments/memory_or_policy_g1_v1.json`（继承 v2.1 `a839ccc592f8aa55`；`frozen: false`，等预检后冻结） |
| 臂（9） | `fh_g fh_m ld_g ld_m mm_g mm_m kb_g kb_m memi` |
| 任务（5） | RouteStick、SwingXtimes、VideoUnmaskSwap、StopCube、VideoRepick |
| episode（val，每任务 5） | RS / SX / SC：0–4；VUS：10 12 13 14 18；VR：10–14 |
| 格数 | 5 × 5 × 9 = **225** |
| 模型 | `openai/gpt-6-luna`，low effort，OpenRouter（v2.1 transport 原样） |
| 观测 | clean；`bridge: memi_named`（MEMI 命名的所有物体都传给 ledger / KB） |
| 上限 | 每 episode 50 次调用、执行器单次 300 s、episode 30 min、reset 600 s；**费用 $2 停止** |
| 用途 | 测每个 P2 对比的 discordance d、检查 prompt 公平、每条终止路径各走一次；**不为任何假设计分**；之后在 NOTES 写带日期的 n |

**VUS 的 episode 说明（orchestrator 决定）：** 10 12 13 14 18 只在主仓库读过目标文本（筛 two-pick 时），从未跑过；
它们都是单目标 goal（blue / red / blue / blue / green，`cl/goals.jsonl`），比被筛走的 two-pick 容易，VUS 的 d 可能偏低。

### 各臂怎么构成（`stress/mop_policy.py`）

- system = MEMI FullHist 的 scaffold + 记忆说明（随记忆变）+ `TASK POLICY` + 策略文本（随策略变）。
- fh：MEMI 的一局一对话历史；ld / mm / kb：每次新请求，MEMI 的决策回合 + `MEMORY` 块 + 回答格式行
  （mm 的格式行多要 `memory_updates`）。
- P-generic：factorial P0（sha256 `7cf07109…`）；P-matched：G0 拆分得到的 `policy_matched.txt`（manifest 里有
  5 个 sha256）。
- ld：`arbiter_kb/mop_memory.py::PreregLedger`；kb：`KBMemory`（冻结 KB 只接 cube / container / target，
  其他类型列在旁边；demo 操作行放在 DEMONSTRATION 下，执行进度只给 execution-only 计数）；mm：MEMI 的 fields，
  模型自己维护；memi：stock MEMI。

## 2. 费用与时间估计

- **费用：** v2.1 每次调用平均 $0.000232（$0.1337 / 577 次）。按 MEMI 专家示范的决策步数（5 任务平均 4.66 次/格）：
  225 × 4.66 × $0.000232 ≈ **$0.24**；mm / memi 的 prompt 约是 FullHist 的 2 倍，调用数也可能翻倍，悲观约 **$1**。
  $2 停止线在每格开始前检查（`enforce_cost_budget`），所以最多超出一格的费用；单格理论上限（50 次调用、每次
  输出到 32k token 上限）约 $0.9，实际 v2.1 最贵一格远低于此。
- **时间（Mac，串行；G0 实测 3 路并行只快 1.07 倍）：** 期望约 **13 h**，悲观约 **27 h**
  （`results/memory_or_policy/g0/summary.json` 的 compute 部分）。建议用 `caffeinate` 在终端里跑。

## 3. 预检要看的东西（已备好）

| 项 | 位置 | 状态 |
|---|---|---|
| 公平 diff（5 任务，每次决策：同一策略在 4 个记忆臂间逐字节相同；同一记忆在 2 个策略间逐字节相同；fresh 臂的决策回合相同；ld / kb 记忆里无建议性用词；memi 臂是 MEMI 原版 POLICY_SYSTEM） | `results/memory_or_policy/g1_prep/fairness.json`，输入 `probe/*.json`（每臂每次决策的完整 prompt） | 通过（见 fairness.json） |
| 脚本化执行器 preflight（经真实 runner、manifest、watchdog） | 已在 Mac 上试跑 5/5 有终态（`results/robomme_common/memory_or_policy_g1/g1_executor_preflight_test/`，未提交） | 通过 |
| gate 脚本 | `scripts/g1_gate.py`：manifest 已冻结、臂/任务/episode/225 格、$2、endpoint probe 返回精确模型、fairness 通过且用的是 manifest 里的 matched 文本、preflight 无阻断、分支干净 | 已写，待冻结后运行 |
| code-reviewer | 第一次 FAIL（0ce4de1），修复见 bec54dd 及之后提交；复审结论见本文 §5 | — |

**probe 的局限：** VUS、VR、RS 的脚本化选择第一步就结束 episode，所以这三个任务只检查了第 1 次决策；多步路径
（执行日志、KB 计数、mm 记忆延续）只在 StopCube（3 步）和 SwingXtimes（4 步）上检查过，另有单元测试。

## 4. 命令（预检通过、你给 go 之后再跑）

在本仓库根目录，Mac 上：

```bash
# 1 冻结 manifest（把 "frozen": false 改成 true），提交，然后登记冻结
.venv/bin/python -c "from pathlib import Path; from arbiter_kb.experiment_manifest import digest; from arbiter_kb import ledger; s=digest(Path('experiments/memory_or_policy_g1_v1.json')); ledger.record('FREEZE', s, 'memory-or-policy G1 pilot manifest; val only; 225 cells; \$2 stop'); print(s)"
```

```bash
# 2 免费的脚本化执行器 preflight（新 run id）
docker run --rm --platform linux/amd64 --network none -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB arbiterkb-robomme-cpu:bb6c37c python stress/common_harness_pod.py --manifest experiments/memory_or_policy_g1_v1.json --stage executor_preflight --memi-root /opt/memi --out /workspace/ArbiterKB/results/robomme_common/memory_or_policy_g1 --run-id g1_preflight_$(date -u +%Y%m%dT%H%MZ)
```

```bash
# 3 endpoint probe（约 $0.000004；key 只从 keychain 进环境，不打印、不落盘）
OPENAI_API_KEY="$(security find-generic-password -s ArbiterKB-OpenRouter -w)" .venv/bin/python scripts/factorial_probe.py --out results/robomme_common/memory_or_policy_g1/endpoint_probe.json
```

```bash
# 4 gate（失败即停）
.venv/bin/python scripts/g1_gate.py --probe results/robomme_common/memory_or_policy_g1/endpoint_probe.json --preflight results/robomme_common/memory_or_policy_g1/<preflight run id> --out results/robomme_common/memory_or_policy_g1/gate.json
```

```bash
# 5 G1 pilot（13–27 h；$2 停止）
export OPENAI_API_KEY="$(security find-generic-password -s ArbiterKB-OpenRouter -w)" OPENAI_API_URL=https://openrouter.ai/api/v1/chat/completions && caffeinate -i docker run --rm --platform linux/amd64 -e OPENAI_API_KEY -e OPENAI_API_URL -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB arbiterkb-robomme-cpu:bb6c37c python stress/common_harness_pod.py --manifest experiments/memory_or_policy_g1_v1.json --stage pilot --memi-root /opt/memi --out /workspace/ArbiterKB/results/robomme_common/memory_or_policy_g1 --run-id g1_$(date -u +%Y%m%dT%H%MZ) --gate-file /workspace/ArbiterKB/results/robomme_common/memory_or_policy_g1/gate.json
```

runner 遇到第一个 infrastructure_error / incomplete 就停（G1 不重跑；预注册的「最多重跑 2 次」用于正式验证）。
中断后用同一 run id 重跑会跳过已完成的格。

## 5. 需要 orchestrator 裁定的两点（未改文本，均为预注册内的冲突）

1. **P-generic 提到「memory snapshot」。** factorial P0 原文：「Use only the current observation interval, the memory
   snapshot, and execution feedback … Do not invent object identity when memory says unknown.」fh 臂没有 snapshot，
   其 scaffold 说「Use the whole history」。文本在各臂间逐字节相同，但含义随记忆变：可能只让 FH-g 吃亏，从而抬高
   H1（matched − generic）和 H3 的分母（MEMI − FH-g）。选项：换成与记忆无关的措辞（新的带日期 NOTES 小节），
   或保持并在每张表注明。
2. **MEMI 的 fields 含观测→菜单字母的映射。** 例如 RouteStick `remaining_path`、SwingXtimes「Use the resulting
   count both to select the next subgoal」。拆分规则 1 要求 fields 原样，所以 mm 臂的记忆说明含建议性内容，和
   「任何记忆不含推荐动作」冲突。fairness 只对 ld / kb 判失败，mm 的建议性用词只计数（每任务 6–32 个）。需要一个
   带日期的裁定，并在 H2（MM-m − LD-m）上注明。

另外，供参考：MEMI 的 subgoal 文本里有两句针对 MEMI **train** episode 的事实（RouteStick-09「B for episode 0,
A for episode 19, …」，SwingXtimes-09「episode 72's first right target …」）。prompt 里不出现 episode 编号，
所以对 val / test 应无直接作用，但说明 MEMI 的策略部分记住了训练集。

## 6. 逐句标注（规则 4，描述性，脚本 + 人工核对）

`results/memory_or_policy/g0/labeling/`：`manual_labels.json`（定义与逐句标签）、
`memi_subgoal_sentence_labels_filled.csv`、`label_shares.json`。主类二选一（choice rule / memory update，后者
含把观测读成状态的感知与估计句），memory reference 是独立标记（句子按名字或部分名字点到 MEMI 字段；只用代词指代的
记在 note 里），因为多数点名字段的句子同时是选择规则。

| 任务 | 句数 | choice rule | memory update | memory reference | 脚本字面匹配字段名 |
|---|---:|---:|---:|---:|---:|
| RouteStick | 12 | 83% | 17% | 17% | 17% |
| SwingXtimes | 16 | 81% | 19% | 19% | 19% |
| VideoUnmaskSwap | 10 | 70% | 30% | 40% | 20% |
| StopCube | 37 | 57% | 43% | 16% | 16% |
| VideoRepick | 23 | 61% | 39% | 30% | 26% |
