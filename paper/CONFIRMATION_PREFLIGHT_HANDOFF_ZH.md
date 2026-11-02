# 「Memory or policy?」正式验证 preflight handoff（未冻结）

日期：2026-11-02

分支：`research/perception-belief-robomme`
交付 commit：以本次提交后 `git rev-parse HEAD` 为准（commit 不能自引用自身 SHA）。

## 状态与硬边界

- **没有读取或运行本仓库的 test episode，没有启动正式验证，没有调用任何付费模型。**
- confirmation manifest 仍为草稿，未写 FREEZE ledger；Mac / pod 及其专属参数仍待 orchestrator / user 决定。
- draft manifest：`experiments/memory_or_policy_confirmation_v1.json`；当前未冻结 hash：
  `9d667f2f043497fb`（完整 sha256 `9d667f2f043497fb19a74013cc1033bc46224b1f434c2f28f6e16fd92777ad6e`）。
- 设计：test 0–49；5 tasks；9 arms；2,250 cells；`openai/gpt-6-luna` low effort；50-call cap；
  总费用停止线 `$6`。
- 当前 draft gate 必须是 NO_GO。`confirmation_preflight_20261102T0952Z` 是在旧 commit 且
  `arbiter_dirty=true` 时得到的开发证据，只可帮助调试 executor，**不得给最终 frozen 包放行**。最终 gate 只接受在
  clean 最终 commit 上重新生成、与最终 manifest hash / commit / sources / params / prompt 完全一致的五行 val 证据。

## 基础设施错误规则

实现位置：

- `arbiter_kb/confirmation_retry.py`：严格分类、同一 cell identity、最多 2 次 rerun、失败尝试留档、全臂 pair drop；
- `stress/common_harness_pod.py`：只在 `confirmation` stage 启用，逐 `(task, episode)` 跨臂核对 seed / reset digest；
- `arbiter_kb/common_supervisor.py`：executor command timeout 归 infrastructure；worker signal / 无结果退出记录为 container crash；
- `tests/test_confirmation_retry.py`、`tests/test_common_watchdog.py`：回归测试。

只有以下六类会 rerun：client 自身 retries 用尽后的 API failure、wrong returned model id、container crash、
executor command timeout、跨臂 reset digest mismatch、跨臂 resolved seed/difficulty mismatch。policy/invalid-answer
failure、task failure、50-call cap、episode wall timeout 都直接保留，不 rerun。

每次 rerun 的 `retry_identity` 固定 manifest、commit/clean、三方 source commits、params/prompt、task/episode/split、
arm、观测配置、model/effort、resolved seed/difficulty 与 factorial inputs；retry history 另持久化第一次 attempt 的
`seed / difficulty / episode_count / reset_digest`，包括 pair 第一个 arm 在内的每次 rerun 都必须逐字段一致，任何变化
立即拒绝。初次 + 2 次完全相同的 rerun 仍失败时，写 `_pair_drops/<task>_ep<episode>.json`，列出并剔除全部 9 arms；
已完成 arm 的原始 row 保留但加 `pair_exclusion`，剩余 arm 不执行，resume 看到 drop record 后继续跳过整个 pair。

定向测试命令与结果：

```bash
.venv/bin/python -m pytest -q \
  tests/test_confirmation_retry.py tests/test_confirmation_gate.py tests/test_common_watchdog.py
```

结果：`21 passed`。全套结果：`317 passed, 6 skipped`；quality `162 files, 0 problems`；self-test
`16 PASS, 0 FAIL, 1 SKIPPED`。覆盖 successful retry、初次 + 2 reruns 后仍失败、9-arm pair drop、
policy/task/cap/episode timeout 不 rerun、wrong model/reset/seed 分类、首臂 rerun reset 改变即拒绝、drop 后剩余 arm
不执行且 resume 整 pair 跳过，以及 stale commit / changed source gate 拒绝。

## StopCube 免费核查

机器可读审计：`paper/STOPCUBE_EXECUTOR_AUDIT.json`。

- 官方固定版本：镜像 `arbiterkb-robomme-cpu:bb6c37c`，MEMI
  `bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0`，`memi/plan/rollout/episode.py:196-234`。
- 官方逻辑：on-menu command 返回 `batch == 0` 时为 `no_op`；官方注释明确举 StopCube last checkpoint 后的
  `remain static` 为例，说明 solve 没有运行 simulator step，下一 call 再显示同一步。
- 本仓库：`stress/common_episode.py:77-93,107-152` 同样把 on-menu zero-batch 记为 `no_op`，0 captured rows 不扩展
  timeline，所以时间不前进。
- 发布的 Luna FullHist / Init / MEMI StopCube 日志做过字段级核对（只看 task/action/time/result/batch/captured，
  不看 goal/observation）：可见 `remain static` → `no_op`、`batch=0`、`captured=0`，下一 timeline range 为空。

**结论：harness 与官方一致，不应修成强制推进仿真时间。** G1 的 t=393 + 重复 `remain static` 是官方 no-op
语义与 policy 选择的交互；不能只凭 G1 现象称为 task dynamics，但源码和发布日志共同支持“没有 harness divergence”。

## 免费 scripted preflight 与真实 prompt fairness

以下命令只用五个预定 val episode，显式断网且清除 API key；不会触碰 test。它必须在 host/frozen manifest/FREEZE
ledger 均提交后的 **clean 最终 commit** 上重新执行；不要复用本次 dirty draft 的 `0952Z` 结果：

```bash
unset OPENAI_API_KEY OPENROUTER_API_KEY
PREP_RUN="confirmation_preflight_$(git rev-parse --short=12 HEAD)"
PREP="results/robomme_common/memory_or_policy_confirmation/preflight_material/$PREP_RUN"
mkdir -p "$PREP/probe"

docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/mop_probe.py --task RouteStick --episode 49 --memi-root /opt/memi \
  --out "/workspace/ArbiterKB/$PREP/probe" --max-decisions 6
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/mop_probe.py --task SwingXtimes --episode 49 --memi-root /opt/memi \
  --out "/workspace/ArbiterKB/$PREP/probe" --max-decisions 6
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/mop_probe.py --task VideoUnmaskSwap --episode 0 --memi-root /opt/memi \
  --out "/workspace/ArbiterKB/$PREP/probe" --max-decisions 6
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/mop_probe.py --task StopCube --episode 49 --memi-root /opt/memi \
  --out "/workspace/ArbiterKB/$PREP/probe" --max-decisions 6
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/mop_probe.py --task VideoRepick --episode 0 --memi-root /opt/memi \
  --out "/workspace/ArbiterKB/$PREP/probe" --max-decisions 6

.venv/bin/python scripts/mop_fairness.py "$PREP/probe" --out "$PREP/fairness.json"

docker run --rm --platform linux/amd64 --network none \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/common_harness_pod.py \
  --manifest experiments/memory_or_policy_confirmation_v1.json \
  --stage executor_preflight --memi-root /opt/memi \
  --out /workspace/ArbiterKB/results/robomme_common/memory_or_policy_confirmation \
  --run-id "$PREP_RUN"
```

本次实跑的真实-request probe fairness 输出为：5/5 tasks、0 fails、`passed: true`。RouteStick 第一次真实请求的
LD-g / LD-m payload 均为 1,334 chars，KB-g / KB-m 均为 2,261 chars；均含实际 cube/stick object records，
不是空 memory。独立 executor preflight 也完成 5/5，全部有 task_failure 类合法终态、0 model calls、5 个 reset
digest 非空；但这些仅是开发证据。最终 gate 会逐行要求精确的 task/episode/split/arm、最终 manifest SHA、当前
Arbiter commit、`arbiter_dirty=false`、pinned sources、params/prompt hashes、0 model calls 和非空 reset digest，并拒绝
任何额外行或重复行。

## Endpoint probe（本次未调用）

这是唯一付费 preflight；遵照本次要求没有执行。它只能在 frozen manifest、FREEZE ledger 和 clean final-commit
免费 preflight 就绪后，由 user 授权运行：

```bash
OPENAI_API_KEY="$(security find-generic-password -s ArbiterKB-OpenRouter -w)" \
.venv/bin/python scripts/factorial_probe.py \
  --model openai/gpt-6-luna \
  --url https://openrouter.ai/api/v1/chat/completions \
  --out results/robomme_common/memory_or_policy_confirmation/endpoint_probe.json
```

gate 要求 `requested_model == returned_model == openai/gpt-6-luna`。

## 最终冻结与放行顺序（现在不要运行）

必须严格按以下顺序，任何后续源码 commit 都会令已有 gate stale：

1. user 选择 Mac 或 pod host 及其专属参数；
2. 写实 frozen manifest 并提交；
3. 对该 manifest hash 记录 FREEZE ledger，并再次提交；
4. 在此 clean 最终 commit 上重新运行上面的免费 val scripted preflight / prompt fairness；
5. 运行 endpoint probe；
6. 生成绑定该 manifest、该 commit 和 source hashes 的 gate；
7. orchestrator 审查 gate 与证据包；
8. user 明确 go 后，才可启动 confirmation。

不得沿用本 draft hash 或 dirty draft preflight。runner 会要求当前 HEAD 精确等于 gate 的 `arbiter_commit`，并在启动前
重新计算 gate 中每个 `source_sha256`；gate 后出现新 commit 或任一受保护源码变化都会 fail closed。

```bash
.venv/bin/python -c "from pathlib import Path; from arbiter_kb.experiment_manifest import digest; from arbiter_kb import ledger; p=Path('experiments/memory_or_policy_confirmation_v1.json'); h=digest(p); ledger.record('FREEZE', h, 'memory-or-policy confirmation; test 0-49 x 5 tasks x 9 arms; 2250 cells; USD 6 stop'); print(h)"

.venv/bin/python scripts/confirmation_gate.py \
  --manifest experiments/memory_or_policy_confirmation_v1.json \
  --probe results/robomme_common/memory_or_policy_confirmation/endpoint_probe.json \
  --probe-dir "$PREP/probe" \
  --fairness "$PREP/fairness.json" \
  --preflight "results/robomme_common/memory_or_policy_confirmation/$PREP_RUN" \
  --stopcube-audit paper/STOPCUBE_EXECUTOR_AUDIT.json \
  --out results/robomme_common/memory_or_policy_confirmation/gate.json
```

## 正式运行命令（现在不要运行）

容器内命令与所有实验 flags 已固定如下；外层 Mac `caffeinate` 或 pod mount/concurrency 只能在环境决定后写入
manifest 并重新 hash/freeze。下面先给 Mac 串行候选，**不是本次对运行环境的决定**：

```bash
RUN_ID=confirmation_20261102T0000Z
OPENAI_API_KEY="$(security find-generic-password -s ArbiterKB-OpenRouter -w)" \
OPENAI_API_URL=https://openrouter.ai/api/v1/chat/completions \
caffeinate -i docker run --rm --platform linux/amd64 \
  -e OPENAI_API_KEY -e OPENAI_API_URL \
  -v "$PWD:/workspace/ArbiterKB" -w /workspace/ArbiterKB \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/common_harness_pod.py \
  --manifest experiments/memory_or_policy_confirmation_v1.json \
  --stage confirmation --memi-root /opt/memi \
  --out /workspace/ArbiterKB/results/robomme_common/memory_or_policy_confirmation \
  --run-id "$RUN_ID" \
  --gate-file /workspace/ArbiterKB/results/robomme_common/memory_or_policy_confirmation/gate.json
```

没有隐藏的 runner flags。model、effort、token/request/phase/episode limits、50-call cap、transport、prices、tasks、arms、
episodes、observation seed 与 retry/drop 规则都来自 frozen manifest，并进入 manifest hash / gate source hashes。

## Code-reviewer 重点

1. `arbiter_kb/confirmation_retry.py`：六类 retry 白名单是否过宽/过窄；首次 pair signature 是否先于 rerun 持久化；
   第三次失败边界及 all-arm exclusion 是否不会被 resume 绕过。
2. `stress/common_harness_pod.py`：gate 的 commit/source-hash 绑定、跨臂 reference 建立顺序、已有结果 resume、pair drop
   后跳过剩余 arms。
3. `arbiter_kb/common_supervisor.py`：executor timeout 从 scored timeout 改成 infrastructure 是否只影响新 protocol 的解释；
   signal / stale log 的 container-crash 分类是否足够严格。
4. `tests/test_confirmation_retry.py`：是否还缺真实 supervisor/result-shape 的 integration case。
5. `experiments/memory_or_policy_confirmation_v1.json` 与 `scripts/confirmation_gate.py`：五行 clean-final-commit provenance、
   无 extra/duplicate、2,250 cell、$6、未决 host、prompt fairness、RouteStick 非空、StopCube pin 是否 fail closed。
