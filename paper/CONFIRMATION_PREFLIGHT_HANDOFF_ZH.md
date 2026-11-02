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
- 当前 draft gate 必须是 NO_GO：实测只报 3 个预期 blocker——未冻结、运行环境待定、工作树有本次未提交改动；
  其余静态 manifest、fairness、RouteStick memory、scripted preflight、StopCube source pin 检查通过。

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
arm、观测配置、model/effort、resolved seed/difficulty 与 factorial inputs；任何变化立即停止。初次 + 2 次完全相同的
rerun 仍失败时，写 `_pair_drops/<task>_ep<episode>.json`，列出并剔除全部 9 arms；已完成 arm 的原始 row 保留，
但加 `pair_exclusion`，未执行 arm 不补跑。

定向测试命令与结果：

```bash
.venv/bin/python -m pytest -q tests/test_confirmation_retry.py tests/test_common_watchdog.py
```

结果：`13 passed`。覆盖 successful retry、初次 + 2 reruns 后仍失败、9-arm pair drop、policy/task/cap/episode
timeout 不 rerun，以及 wrong model/reset/seed 分类。

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

以下命令只用 val episode，显式断网且拒绝 API key；不会触碰 test：

```bash
unset OPENAI_API_KEY OPENROUTER_API_KEY
PREP=results/robomme_common/memory_or_policy_confirmation/preflight_material
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
  --run-id confirmation_preflight_20261102T0952Z
```

本次实跑的真实-request probe fairness 输出为：5/5 tasks、0 fails、`passed: true`。RouteStick 第一次真实请求的
LD-g / LD-m payload 均为 1,334 chars，KB-g / KB-m 均为 2,261 chars；均含实际 cube/stick object records，
不是空 memory。独立 executor preflight 也完成 5/5，全部有 task_failure 类合法终态、0 model calls、5 个 reset
digest 非空；结果在 `results/robomme_common/memory_or_policy_confirmation/`。`scripts/confirmation_gate.py` 会重新检查
这些原始文件，不接受只看摘要。

## Endpoint probe（本次未调用）

这是唯一付费 preflight；遵照本次要求没有执行。正式冻结前由 user 授权后运行：

```bash
OPENAI_API_KEY="$(security find-generic-password -s ArbiterKB-OpenRouter -w)" \
.venv/bin/python scripts/factorial_probe.py \
  --model openai/gpt-6-luna \
  --url https://openrouter.ai/api/v1/chat/completions \
  --out results/robomme_common/memory_or_policy_confirmation/endpoint_probe.json
```

gate 要求 `requested_model == returned_model == openai/gpt-6-luna`。

## 冻结与 gate（现在不要运行）

先由 orchestrator / user 决定 Mac 或 pod，把 manifest 的 `execution_host`、`environment_selection` 和最终专属 flags
写实，再把 `frozen` 改为 `true`、提交；随后记录 FREEZE 并单独提交 `results/test_reads.log`。不得沿用本 draft hash。

```bash
.venv/bin/python -c "from pathlib import Path; from arbiter_kb.experiment_manifest import digest; from arbiter_kb import ledger; p=Path('experiments/memory_or_policy_confirmation_v1.json'); h=digest(p); ledger.record('FREEZE', h, 'memory-or-policy confirmation; test 0-49 x 5 tasks x 9 arms; 2250 cells; USD 6 stop'); print(h)"

.venv/bin/python scripts/confirmation_gate.py \
  --manifest experiments/memory_or_policy_confirmation_v1.json \
  --probe results/robomme_common/memory_or_policy_confirmation/endpoint_probe.json \
  --probe-dir results/robomme_common/memory_or_policy_confirmation/preflight_material/probe \
  --fairness results/robomme_common/memory_or_policy_confirmation/preflight_material/fairness.json \
  --preflight results/robomme_common/memory_or_policy_confirmation/confirmation_preflight_20261102T0952Z \
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

1. `arbiter_kb/confirmation_retry.py`：六类 retry 白名单是否过宽/过窄；resume 时 history/identity；第三次失败的边界；
   all-arm exclusion 是否不会被后续 resume 绕过。
2. `stress/common_harness_pod.py`：跨臂 reference 建立顺序、seed/reset mismatch、已有结果 resume、pair drop 后跳过剩余 arms。
3. `arbiter_kb/common_supervisor.py`：executor timeout 从 scored timeout 改成 infrastructure 是否只影响新 protocol 的解释；
   signal / stale log 的 container-crash 分类是否足够严格。
4. `tests/test_confirmation_retry.py`：是否还缺真实 supervisor/result-shape 的 integration case。
5. `experiments/memory_or_policy_confirmation_v1.json` 与 `scripts/confirmation_gate.py`：2,250 cell、$6、未决 host、
   provenance、prompt fairness、RouteStick 非空、StopCube pin 是否 fail closed。
