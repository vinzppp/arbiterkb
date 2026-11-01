# RoboMME Canonical Memory 研究协议（v2 执行版）

状态：**Step 0 已执行；Track A 窄通过，Track B / scored pilot gate 未通过**

分支：`research/perception-belief-robomme`

分析源码提交：`a04a193`

`main` 固定：`069beb7688fe1b853dd1fcc09ba6e3e481372f87`

## 1. 研究问题

在相同观察、任务语义和下游决策下，概率化、可 retract 的 object memory 是否比强 deterministic
ledger 提供增量价值？本协议把问题分成两条互不替代的证据链：

1. **Track A — state estimation**：identity / cover state、unknown、proper score、recovery 是否更好；
2. **Track B — decision utility**：状态差异是否实际改变合法动作，并进一步改变闭环成功。

Track A 的改善不能自动外推为 Track B。deterministic probe 只是隔离 memory intervention 的实验仪器，
不是通用 controller 的性能证明。

## 2. 冻结范围与禁区

- 仅使用已有 G-train、development 记录和既有 RoboMME validation/factorial traces。
- G-train episode 文件名必须以 `ep0`–`ep9` 结尾；脚本对 `>=10` fail closed。
- 不打开、不复用 held-out G-test 内容；只允许引用既有报告中的 aggregate conclusion。
- 不启动新 simulator rollout，不调用 LLM，不读取 credential，不产生 API 费用。
- 不运行 observation degradation，不选择性重跑，不改旧结果。
- 不增加 KB rule，不建设完整 controller，除非 Step 0 的 action-identifiability 与 novelty gate 都通过。
- `main` 只读验证；任何修复仅在研究分支实现，且不得重写冻结轨迹。

## 3. Step 0：零成本可辨识性审计

### 3.1 相同输入与比较臂

相同 relinked G-train perception stream 同时送入：

- `strong_ledger`：现有 causal relinked source + `TrackerSystem("bound")`；
- `arbiterkb`：冻结规则与概率 belief；
- `factorial_ledger`：此前 pilot 的 minimal deterministic ledger，仅作为 adapter 诊断，不作为最强主基线。

ground truth 只在 memory 完成输出后用于评分和 tracker-error 统计，不进入 canonical state 或动作 probe。

### 3.2 最小 canonical projection

只投影 identity/cover 决策所需字段：

- target object；
- 完整 container candidate space；
- 显式 `unknown/elsewhere`；
- 每个候选概率；
- 当前可见 candidate 的授权 pixel point；
- evidence time、support 与 provenance。

projection 不补入 simulator identity、oracle pose、不可见 point 或额外概率质量。候选空间缺失的概率为 0，
总质量归一化；完全无质量时只把 `unknown` 设为 1。

### 3.3 决策 probe

- **hard probe**：选择 top-1 且当前可见的 candidate；否则返回“无可辩护 pick”。
- **threshold probe**：在 hard probe 上加冻结的 `tau_hi=0.8`；低于阈值时只能报告 blocked。

blocked 不是 RoboMME 的合法 re-observe / verify 动作，不能计为 controller 改善。只有双方都能发出合法 pick
且 target 不同，才是最强 action-relevant disagreement。

### 3.4 冻结指标

- top-1 / cover-relation disagreement；
- unknown disagreement；
- hard 与 threshold actionability disagreement；
- 双方合法但 pick target 不同；
- top-1 accuracy、Brier、log loss、ECE、`p_max`；
- paired episode-cluster bootstrap（G-train episode 为 resampling unit，10,000 draws，seed `20261101`）；
- tracker pass swap、missed detection、false association、recovery、occlusion duration；
- 已记录 factorial 决策中的 normalized/legacy progress conflict。

这些 G-train 指标是当前规则开发使用过的 in-sample diagnostic，不能当 independent validation。

## 4. Gate 规则与本次执行决定

### 4.1 Track A hard-state gate

若同一观察流上出现非选择性的 identity/unknown disagreement，且至少一个 fixture 或真实记录能令相同 probe
选择不同合法 target，则“状态可改变动作”这一必要条件通过。

### 4.2 Track B calibration gate

若 RoboMME 动作空间没有合法的 verify、re-observe、defer 或 recovery action，则 confidence/calibration 只能阻止
动作，不能产生可比较的合法替代动作。此时 calibration-to-success 不可辨识，不得运行以此为 claim 的 scored pilot。

### 4.3 本次结论

- Track A hard-state：**窄通过**；存在 4 个双方均合法但 pick target 不同的 G-train 查询。
- Track B calibration：**失败**；现有 `multi_choice` 不支持合法的信息收集/恢复动作。
- full canonical controller：**不建设**；离线 probe 已足够回答 Step 0，继续建设会增加没有被识别的复杂度。
- 最多 12-cell scored pilot：**不执行**；没有净 top-1 增益、G-train 为 in-sample、action mediation 太稀疏，
  且 calibration utility 不可辨识。
- held-out 与 degradation：**保持关闭**。

## 5. 若未来重新开启 scored pilot，必须重新预注册

只有新的、未用于规则开发的 development perception episodes 显示稳定 action-relevant disagreement，并且任务提供
合法的 uncertainty-sensitive action，才可重新冻结：

- 2 tasks × 3 paired episodes × 2 memory arms，最多 12 cells；
- 相同 raw observations、canonical schema、controller bytes、seed、reset、executor、watchdog；
- primary Track A metric 与 primary Track B paired success；
- complete candidate space 与 unknown；
- paired episode analysis、multiplicity family、10pp practical-effect threshold；
- API 上限 `$0.10`，无新 paid GPU/cloud，无 selective rerun；
- 所有 task failure、timeout、infrastructure error 原样保留。

本次没有生成 pilot 报告，因为 scored pilot 没有执行。

## 6. 可复现入口

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/canonical_identifiability.py \
  --out results/robomme_common/canonical_memory_v2/identifiability

.venv/bin/python scripts/canonical_timing.py \
  --out results/robomme_common/canonical_memory_v2/timing_comparison.json

.venv/bin/python -m pytest -q
```

机器可读输出：

- `results/robomme_common/canonical_memory_v2/identifiability/summary.json`
- `results/robomme_common/canonical_memory_v2/identifiability/rows.json`
- `results/robomme_common/canonical_memory_v2/identifiability/reproducibility_manifest.json`
- `results/robomme_common/canonical_memory_v2/timing_comparison.json`

## 7. 安全与研究完整性声明

本轮新 simulator rollouts = 0、LLM calls = 0、API cost = `$0.00`、held-out records opened = false、
degradation run = false、selective reruns = false。OpenRouter credential 未被读取、打印、写盘或提交。

