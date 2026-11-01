> **STOPPED（2026-11-01，orchestrator）：** Phase 0 可行性结论 INCONCLUSIVE；Stage B 由 orchestrator 停止（无统计功效：30 个 episode 预计约 1 次与动作相关的分歧）。本实验不再继续；论文方向见主仓库 docs/HANDOFF_ORCHESTRATOR.md「Paper direction」。

# ArbiterKB：Fresh Tracker-Error Replay 可行性与研究报告

日期：2026-11-01  
分支：`research/perception-belief-robomme`  
Phase 0 source HEAD：`a982981b9f4b69abf0f4cc4ee45508b39a850dd5`  
`main`：`069beb7688fe1b853dd1fcc09ba6e3e481372f87`（未修改）  
研究判定：**INCONCLUSIVE — Phase 0 停止，未进行比较评分**

## 摘要

本轮没有得到合格的 fresh visual tracker traces。仓库内两类看似可用的数据都不满足 prospective gate：

1. G 有真实 GroundingDINO/SAM/Cutie 输出，但 0–9 已用于规则开发，10–19 是禁止打开的 held-out；
2. 本地 RoboMME LeRobot 影像来自公开 revision `92796abb…`，metadata 只声明 `train: 0:1600`，本仓库又已
   对相同 episode 做过 frame、VLM、robot-state 与 geometry 分析。它不是 fresh batch，且没有 depth、mask、
   object pose 或 camera calibration 供当前 perception/memory adapter 做无特权重放。

当前 host 与既有 CPU container 也无法运行冻结 perception：host/project environment 没有 `sg_pilot`、
GroundingDINO、SAM、Cutie 或 CUDA；`arbiterkb-robomme-cpu:bb6c37c` 仅有 CPU OpenCV，不含上述组件，并存在
amd64 image / arm64 host 的 platform mismatch。旧 pod 只留下 perceived/results，不存在可重新选择的独立
fresh raw RGB-D batch。

因此按协议停止：没有运行 memory replay、没有生成比较 outcome、没有访问 held-out、没有启动 simulator、
没有 API/LLM/GPU 花费。**未观察到差异不等于方法相同；零错误计数也不等于零错误率。**

## 1. Phase 0 审计

| Gate | 结果 | 证据与解释 |
|---|---:|---|
| fresh unseen visual material | FAIL | G-train 已开发使用；RoboMME public train 已多次使用；`cl_pod` 是旧结果 |
| actual frozen detector/tracker runnable | FAIL | host 与现有 CPU image 都缺少 `sg_pilot`/DINO/SAM/Cutie；无 CUDA |
| legitimate nonleaking GT | FAIL | 没有新录制的独立 identity GT；公开 RoboMME 表不含 object identity/masks/poses |
| zero prior overlap | FAIL | 所有本地可见视觉候选均为旧 G 或旧 public RoboMME episode |
| enough natural errors | 未知/FAIL | 没有 eligible trace，因此不能测 prevalence 或 recovery |

机器审计只列了 G 文件名：180 files，其中 development suffix 0–9 共 90、held-out suffix 10–19 共 90；
held-out payload opened = false。公开 RoboMME metadata 显示 1,600 train episodes；本地转出 500 episode npz，
分属 BinFill、PickXtimes、SwingXtimes、VideoPlaceOrder、VideoRepick 各 100；另有 200 extracted-frame npz 与
6 个 packed mp4。episode/video payload 在本轮均未打开。

`data/cl_pod` 有 148 个历史 `result.json`，但没有满足本研究要求的 fresh、paired、byte-identical raw common
observation traces。这些记录不能补做 prospective Track A/B。

## 2. 为什么没有“退而求其次”运行

- 使用 G-train 会把规则开发数据当独立验证；
- 打开 G-test 会违反明确 held-out 禁令；
- 使用 public RoboMME train 会复用先前已检查的 episode，而且现有 tables 只提供 images、robot state、actions
  与 subgoal text，没有合法 identity GT；
- 人工 swap track ID 会重新引入“为规则量身定做 corruption”的 circularity；
- 用 simulator object IDs/text 替代像素 tracker output 不回答 primary question；
- 仅安装一套本地模型也不能创造 fresh scene、independent GT 或 zero overlap。

所以本次停止是 data/provenance gate 的结果，不是工程失败后把 missing result 写成负结果。

## 3. Track A — memory estimation

未运行。eligible independent episodes = 0，eligible decision points = 0，measured tracker errors = 0（含义是
“未测量”，不是“没有错误”）。ledger 与 ArbiterKB 的 top-1、Brier、log loss、recovery time 均为 `null`。

历史 G-train 的 241 queries 结果仍只作背景：两者 top-1 均 83.40%，13 次分歧、4 次双方都合法但 target
不同、各 3 次 unique win，只有 1 次分歧落在 measured pass-swap。它不是本轮 prospective evidence，不能
填入 fresh result table。

## 4. Track B — decision utility

未运行。没有 common observation trace，就不能构造公平的 paired memory states；也没有新 GT 可以评价
correct legal pick。primary paired effect、action disagreement、wrong-target rate、regret 与 clustered CI 全部
为 `null`。

冻结 acquisition protocol 已明确未来使用 existing canonical legal-pick projection，并在 scoring 前要求三个
fixture：同 state 同 action、arm label 不影响 action、posterior 排序可改变一个合法 pick。接口仍不新增
verify/re-observe/abstain。

## 5. Precision 与样本量

计划的 30 independent episodes 不是充分把握的 confirmatory study。对 episode-level paired difference，若
discordance variance proxy 为 0.30，n=30 的正态近似 95% half-width 为 19.6 pp；要达到 10 pp half-width
约需 116 episode。最坏方差下为 35.8 pp 与 385 episode。

因此未来 30-episode Stage B 的作用是检测大效应、测 error prevalence 与决定是否值得扩样；不能用一个跨
零区间声称 equivalence，也不能把 repeated frames 当独立样本扩大 n。

## 6. 已冻结的可执行 acquisition plan

`experiments/fresh_tracker_error_acquisition_v1.json` 已固定：

- 6 个 Stage A technical episodes，随后 30 个 prospective development episodes；
- 三类 controlled scenario 各 10 个：single occlusion、single close pass/swap、repeated occlusion/swap；
- 每 episode 两个 identity-critical legal-pick point；
- 固定 calibrated RGB-D task camera、独立隐藏 GT stream、两位 blind annotators；
- raw stream write-once + SHA-256、exact/perceptual duplicate checks；
- open task-derived detector vocabulary，禁止 simulator scene count/colour/identity；
- 同一 serialized trace byte-identically 输入 ledger 与 ArbiterKB；
- technical exclusions 在比较前签定，禁止按两臂 outcome 筛 episode。

Stage A 只有在 6/6 可处理、annotation 通过、零 leakage，且至少出现一个自然产生的 association 或
occlusion-recovery error 时才通过。详细规则见
`paper/FRESH_TRACKER_ERROR_REPLAY_PROTOCOL_ZH.md`。

## 7. Runtime、成本与完整性

| 项目 | 数值 |
|---|---:|
| new visual/perception episodes | 0 |
| new memory replays | 0 |
| simulator/scored rollouts | 0 |
| LLM/API calls | 0 |
| API cost | `$0.00` |
| paid GPU hours | 0 |
| held-out payload opened | false |
| degradation | 未运行 |

Phase 0 audit 脚本 fail-closed 检查 branch 与 `main` ref。机器输出：

- `phase0_feasibility_audit.json`
- `perception_tracking_error_stats.json`
- `paired_memory_action_comparison.json`
- `episode_cluster_uncertainty.json`
- `runtime_cost.json`
- `reproducibility_manifest.json`
- `validation.json`

均位于 `results/robomme_common/fresh_tracker_error_replay_v1/`。credential scan 与完整 test/quality 结果在本次
commit 前执行；没有写入或调用任何 credential。

## 8. Research decision

**INCONCLUSIVE。** 当前 evidence 既不支持 GO，也不支持对 full ArbiterKB 作新的 NO-GO：关键 prospective
comparison 根本没有发生。历史结果仍令 full KB 处于必须先击败 strong ledger 的位置，但不能用数据缺失把
它宣布等价或无效。

### 8.1 与此前 empirical NO-GO 的区别

这是两个不同层次的判断，不能合并：

- **此前 NO-GO 是 empirical architecture gate**：在已经执行的 v2.1/factorial/canonical development
  evidence 中，full ArbiterKB 没有击败 simpler ledger/FSM，部分 KB state/progress 还引入错误。因此当时有
  数据支持“不继续扩大 full architecture validation”。
- **本轮 INCONCLUSIVE 是 Phase 0 data/infrastructure gate**：没有新的独立 RGB-D + GT batch，也没有可运行的
  pinned tracker runtime，所以 ledger-vs-Arbiter prospective hypothesis 根本没有被检验。本轮不是新的性能
  失败，也没有减弱此前 NO-GO；它只说明目前不能用 fresh evidence 更新那个判断。

研究管理上的合并结论是：full KB 继续冻结；理由仍是此前 empirical NO-GO，而不是本轮的 infrastructure
failure。若未来 Stage A/Stage B 产生强的 action-relevant advantage，才有依据重新打开 architecture gate。

### 8.2 未来 prospective study 的必要条件（只列要求，不执行）

Data：

- 6 个 Stage A + 30 个 Stage B 的新录制 episode，全部在本 manifest 冻结后产生；
- 每个 episode 有同步 RGB、aligned depth、timestamps、intrinsics、extrinsics 和不可见于 task RGB 的 identity GT；
- 三个相同 container 的 occlusion/close-pass/swap 场景按固定 ID/schedule 全量保留；
- raw streams write-once、逐文件 SHA-256，并与历史允许范围内的 media 做 exact/perceptual duplicate audit；
- 两名不看 memory outcome 的 annotator，decision point 全部 adjudicate；
- 明确许可与保存期限；G-test 永远不参与 acquisition、duplicate check 或调参。

Infrastructure：

- 能运行 pinned `sg_pilot`、GroundingDINO、SAM、Cutie 的受控环境及其 commit/weight hashes；
- 兼容的本地 GPU/CUDA/driver 或另行批准的 GPU 资源；当前 CPU image 不合格；
- RGB-D recording/calibration 工具与独立 GT camera/hidden-marker channel；
- 足够空间保存 raw、mask、common trace、annotation 和 hash manifest；
- Stage A 后冻结 vocabulary、threshold、tracker/relink config，并产生一份两个 arms byte-identical 的 trace；
- scoring machine 不持有 ground truth，直到 memory states 与 legal actions 都完成 content hash；
- 预先通过 action-sensitivity fixtures、episode-cluster analysis 与 technical-exclusion audit。

此时不扩 KB、不做 held-out、不做 observation degradation，也不做更多旧 episode LLM rollout。最省信息
成本的下一步是执行 manifest 中的 6-episode Stage A：用新录制 RGB-D + 隐藏 GT 跑冻结 tracker，先回答
“是否真的能稳定产生、分类并评分 tracker error”。若 Stage A 失败，停止该机制方向；若通过，才收满 30
episode 并运行 paired replay。

## 9. 最终九问

1. **是否得到真正 fresh visual tracker traces？** 没有。
2. **有多少独立 episode 与 measured tracker errors？** eligible episode 0；measured error 0，意为未测量。
3. **ledger 与 ArbiterKB 的 state estimation 是否不同？** 本轮未知；没有运行两臂。
4. **差异是否改变正确/错误 legal action？** 本轮未知；没有可评分 action。
5. **ArbiterKB 是否在预注册 primary outcome 胜出？** 没有结果，不能判断。
6. **是否有 identity switch/occlusion recovery 的局部 gain？** 没有 eligible subgroup 数据。
7. **结果是否具有统计或实际意义？** 当前只有可行性 blocker，效应 estimate 与 CI 均不存在。
8. **full KB 的复杂度是否合理？** 现有 prospective evidence 仍不足以证明合理；保持冻结，不扩展也不删除。
9. **下一项最有效实验？** 只做 6 个新 RGB-D controlled episodes 的 Stage A tracker/GT 技术采集；通过后才做
   30-episode paired ledger-vs-Arbiter replay。
