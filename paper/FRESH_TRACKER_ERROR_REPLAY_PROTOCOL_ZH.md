# ArbiterKB：Fresh Tracker-Error Replay 与 Memory Utility 冻结协议

状态：**v1 已冻结，等待新采集；当前不得进入比较评分**  
冻结日期：2026-11-01  
acquisition manifest：`experiments/fresh_tracker_error_acquisition_v1.json`  
manifest canonical SHA-256：`439b7497e498339736d822c1a3a08313272a0e9aaa77120082b0fb1d42e97ba7`

## 1. 研究问题与边界

主要问题是：在**新采集、此前未见、由真实视觉处理链自然产生的 tracker error** 下，冻结的 ArbiterKB
是否比强 deterministic ledger 更少产生会改变合法 pick 的 identity error？

本研究严格分为：

- Track A — memory estimation：同一 observation trace 下的 identity top-1、Brier、log loss、错误持续与恢复；
- Track B — decision utility：同一 deterministic legal-pick rule 下的正确目标率、错误目标率、paired action
  disagreement 与 regret。

本研究是 offline replay，不声称闭环任务成功；不增加 KB rules，不修改 architecture，不运行 RoboMME scored
rollout，不访问 held-out，不注入合成 tracking error。

## 2. Phase 0 fail-closed gate

比较评分前必须同时满足：

1. 有独立于 G-train/G-test、公开 RoboMME 旧实验和 `cl_pod` 旧运行的新视觉记录；
2. 冻结 GroundingDINO → SAM → Cutie 链实际处理像素，而不是读取 simulator identity、object pose 或文本；
3. identity ground truth 来自 memory 不可见的独立来源，且只在 memory/action 序列化后 join；
4. 有可审核的 camera、scene、object、tracker、annotation provenance；
5. Stage A 中至少测到一个 association 或 occlusion-recovery error，证明数据能回答机制问题。

任一条件失败即停止。零 eligible episode 必须记录为“未运行”，不能把 error count 的零解释为零错误率。

## 3. 既有数据排除规则

- `data/g_perceived` 的 0–9 已用于规则开发，不能作为 fresh evidence；10–19 是 held-out，禁止打开；
- Hugging Face RoboMME revision `92796abb…` 的 metadata 声明唯一 split 为 `train: 0:1600`，本仓库已经用其
  episode 做 robot-state、frame extraction、VLM 和 geometry 分析，不能重新包装为 prospective batch；
- `data/cl_pod` 是既往 controller/perception 结果，不是新的 paired raw observation stream；
- 任何 simulator-generated text observation、oracle identity 或人工 synthetic id swap 均不合格。

Phase 0 只允许读取文件名、公开 source/split metadata 和已有报告。held-out payload 不读取、不 hash。

## 4. Prospective acquisition

### 4.1 两阶段采集

- Stage A：`FTER-A-001`–`006`，6 个技术 episode，只验证 recording、tracker、annotation 和 leakage；不进入
  memory comparison。
- Stage B：`FTER-B-001`–`030`，30 个独立 development episode；每个固定两个 identity-critical decision
  point，共预计 60 个 decision。episode 是统计独立单位。

所有 episode 必须按 ID 顺序录制并保留。不能因为没有 tracker error、ledger 正确、Arbiter 错误或 effect
变小而删除或补录。

### 4.2 Scene 与 camera

使用三个视觉相同的不透明 container、一个 target item、一个 distractor 和固定 legal-pick menu。三类受控
场景各 10 个 Stage B episode：

- 单次遮挡；
- 单次 close pass / swap；
- 重复遮挡与 swap。

这类数据的标签是 **controlled scenarios with naturally occurring tracker errors**：动作被刻意安排，但错误必须
由冻结视觉 pipeline 自然产生，不能手工篡改 track ID。

task camera 必须固定、校准并同步记录 RGB-D（至少 640×480、10 Hz）、intrinsics、extrinsics 与 timestamp。
两个预注册机位按 episode 奇偶交替；背景按固定 schedule 取 plain-light、plain-dark、moderate-clutter。

### 4.3 Ground truth

使用独立 overhead camera 或 task RGB 不可见的隐藏 identity marker。两位 annotator 独立标注 decision point、
遮挡前后及 close pass 两侧的 identity；分歧在不看两臂输出的条件下 adjudicate。decision point adjudication
后必须 100% 一致，adjudication 前 identity agreement 至少 0.95。

ground truth、未来 frame、隐藏 marker 与第二机位不得进入 detector、tracker、ledger、Arbiter 或动作规则。

### 4.4 技术排除

唯一允许排除：required stream 损坏、相机离开校准位、timestamp drift 超过一帧、ground truth 无法裁决、
脚本没有完成。排除表必须在比较 scoring 前签定。

原始 stream 写入后立即 SHA-256；对历史允许范围内的 development media 做 exact hash 与 perceptual hash
duplicate audit。G-test 不得为了 duplicate check 而打开。

## 5. Perception freeze

复用 `stress/perceive.py` 的 pinned GroundingDINO → SAM → Cutie adapter。Stage A 后、任何 Stage B memory output
生成前，冻结：代码 commit、所有 weights hash、Python/CUDA/driver、detector phrase、confidence threshold、mask
threshold、tracker config、relink config、occlusion threshold 与 calibration。

只能使用由 task 得到的 open vocabulary；禁止 `--vocab scene` 中由 simulator scene 泄漏的 object count、颜色
或类别。每个 episode 输出一次 serialized common observation trace，两个 memory arm 读取同一 byte hash。

每帧保存 track ID、candidate class、bbox/mask reference、detection confidence、missing interval、association、
reappearance 与 timestamp。错误分成 detector failure、tracker association failure、occlusion ambiguity、genuine
identity ambiguity 与 memory update error。

## 6. 两个冻结 memory arms

只比较：

- A：现有 strong deterministic ledger，保留 causal relink、unknown 与 recovery；
- B：现有 ArbiterKB，保留 probabilistic identity belief、evidence update 与 retraction。

二者必须共享 observation bytes、clock、task info、demo/execution segmentation 与 task-progress layer。arm 名称不得
进入任何 update 或 decision branch；不得按 fresh outcomes 调任一 arm。

## 7. Legal-pick decision contract

使用已有 canonical legal-pick projection：对当前可见的 legal targets，选择 identity mass 最大者；完全相同则按
冻结 menu order tie-break。不存在 abstain、verify、re-observe 等接口中没有的动作。

评分前运行三类 fixture：

1. 相同 canonical states 必须产生相同动作；
2. 仅更改 arm label 不得改变动作；
3. 一个有效的 posterior 排序变化必须能改变 available legal pick。

若第 3 项失败，Track B 因 action set 无 uncertainty sensitivity 而停止，不能把 calibration 包装为 utility。

## 8. 统计冻结

Primary outcome：全部 eligible decision points 的 paired correct-target legal-action rate，按 episode cluster 计算
ArbiterKB − ledger risk difference。episode 是 bootstrap 与推断单位，固定 10,000 draws、seed `20261101`。

minimum practically meaningful effect 为 +10 pp。primary 不做 multiplicity adjustment；冻结 secondary family
使用 Holm：wrong-target rate、action regret、identity top-1、Brier、log loss、error persistence、recovery time。
error-type subgroup只作 secondary；局部 subgroup gain 不足以判 GO。

unknown、missing、非法或无法映射的 action 在 primary 中计错，并另报原因。系统性 trace corruption 使整个
comparison invalid，而不是选择性 rerun 某 cell。

30 episode 只是 prospective development batch，不是已证明充分的 confirmatory sample。若 paired discordance
约 0.30，episode-level difference 的正态近似 95% half-width 在 n=30 时约 19.6 pp；达到 10 pp half-width 约需
116 episode。最坏方差下分别约 35.8 pp 与 385 episode。因此 n=30 只能发现大信号并估计 error prevalence。

## 9. Gate 与解释

- GO：fresh measured errors 上 state 与 legal action 同时出现可重复、实际重要的 Arbiter 增益，且不是单一
  subgroup 或 annotation artifact；
- NO-GO：数据与动作敏感度充分，但 strong ledger 仍相当或更好，full KB 复杂度不值；
- INCONCLUSIVE：fresh data、真实 tracker errors、ground truth、action sensitivity 或 precision 任一不足。

只有 Stage A 全部 required streams 合格、6/6 tracker 完成、annotation 通过、零 leakage，且至少一个可测
association/recovery error，才允许 Stage B。Stage B 后也不得自动打开 held-out。

## 10. 成本与复现

本研究不需要 LLM。API 上限 `$0.10`；未经单独授权禁止 RunPod、paid GPU 或 cloud rental。当前复现入口：

```bash
PYTHONPATH=. .venv/bin/python scripts/fresh_tracker_feasibility.py
```

Phase 0 outputs 位于 `results/robomme_common/fresh_tracker_error_replay_v1/`。在真正采集后，必须用新 commit
追加 raw stream hashes、pipeline/weights hashes、annotation manifest 和 common-trace hashes；不得覆盖本次
blocked audit。
