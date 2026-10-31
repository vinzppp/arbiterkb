# ArbiterKB R2.1：RoboMME 四臂冻结 validation 报告

日期：2026-10-31（America/Los_Angeles）

分支：`research/perception-belief-robomme`；结果提交：`d2ba092`；`main` 保持在
`069beb7`，未修改。

## 结论先行

冻结 v2.1 的四任务 × 四臂 smoke 和后续 9 episodes/task clean validation 已真实
完成。成功 validation 的 run ID 是 `validation_20261031T150046Z`：144/144 cells
都有协议允许的终态，审计 PASS；无 infrastructure error、incomplete、harness
error、wrong model、非法调用、dirty provenance 或 reset/seed mismatch。5 个 episode
watchdog timeout 均按预注册 intention-to-treat 规则计为失败，没有选择性重跑。

研究结果是负面的。MEMI 取得 24/36（66.7%），FullHist 17/36（47.2%），完整
ArbiterKB 15/36（41.7%），soft identity repair 9/36（25.0%）。完整 ArbiterKB
没有超过简单 FullHist，更没有超过 MEMI；在 VideoRepick 上是 0/9。六个冻结总体
contrast 中，Holm 校正后只有 MEMI 优于 soft 显著（`p=0.00165`）。

因此闸门分成两层：

- **GO（实验可靠性）**：common harness、watchdog、固定 reset、模型身份、轨迹和成本
  记录足以支持这组 validation 结论。
- **NO-GO（继续扩大）**：不进入 observation degradation，不读取 held-out test。
  当前证据不支持为 ArbiterKB 再消耗测试集；先停止规则扩展并重新判断研究假设。

## 冻结协议与运行链

Manifest SHA 为 `a839ccc592f8aa55`，四臂共用 `openai/gpt-6-luna`、OpenRouter
chat-completions endpoint、low effort、相同 val episodes 1--9、相同 clean observation、
官方 `multi_choice` executor、每 cell 新进程和固定 watchdog。MEMI、RoboMME、
ManiSkill 分别固定在 `bb6c37c`、`fae9f73`、`07be6fb`；CPU 镜像 ID 为
`sha256:07424f3b53e08c0bfd27db3a97d89210de92679b040aed3a5aa49e580e059fd1`。

新授权 credential 只从 macOS Keychain 注入进程环境；此前聊天中暴露的 key 没有
读取或恢复。冻结 endpoint 的最小 probe 返回 HTTP 200 和精确模型
`openai/gpt-6-luna`，15 tokens、成本 `$0.0000035`。仓库与结果中没有 credential。

完整链条保留了所有失败：

1. `smoke_20261031T142402Z`：16/16，9 success、7 task failure，audit PASS。
2. `validation_20261031T145940Z`：父进程没有把 gate file 传给 worker，首格在 API
   前成为 infrastructure error；成本 `$0`。该 run 原样保留，未合并、未补格。
3. 只修复 gate-file 参数传递并加回归测试后，以新 run ID 启动
   `validation_20261031T150046Z`。所有 144 格来自这一 run。

## 主要成功率

括号为逐 arm 的 Wilson 95% 区间。

| arm | success | 95% CI | timeout | 平均调用/格 | 平均墙钟/格 |
|---|---:|---:|---:|---:|---:|
| FullHist | 17/36 = 47.2% | 32.0%--63.0% | 0 | 2.44 | 111.7s |
| MEMI | 24/36 = 66.7% | 50.3%--79.8% | 1 | 3.58 | 145.4s |
| soft repair | 9/36 = 25.0% | 13.8%--41.1% | 4 | 6.83 | 234.2s |
| ArbiterKB | 15/36 = 41.7% | 27.1%--57.8% | 0 | 3.17 | 127.4s |

按任务分解：

| task | FullHist | MEMI | soft | ArbiterKB |
|---|---:|---:|---:|---:|
| VideoUnmaskSwap | 3/9 | 7/9 | 5/9 | 7/9 |
| ButtonUnmaskSwap | 3/9 | 3/9 | 0/9 | 2/9 |
| PickHighlight | 9/9 | 9/9 | 4/9 | 6/9 |
| VideoRepick | 2/9 | 5/9 | 0/9 | 0/9 |

ArbiterKB 在 VideoUnmaskSwap 与 MEMI 同为 7/9，但该优势没有跨任务稳定：它在
PickHighlight 落后于 FullHist/MEMI，在 VideoRepick 全败。完整 KB 的总体点估计比
FullHist 低 5.6 percentage points，也没有显示复杂 belief machinery 的净收益。

## 配对统计与不确定性

风险差为前者减后者；区间是按同一 task/episode 成对、10,000 次确定性 bootstrap
的未校正 95% percentile interval。`p_Holm` 是六个冻结 contrast 内的 exact paired
McNemar 多重校正值；正式显著性结论以 `p_Holm` 为准。

| contrast | success risk difference | 95% paired bootstrap | discordant（前胜/后胜） | p_Holm |
|---|---:|---:|---:|---:|
| MEMI − FullHist | +19.4pp | 0.0pp--36.1pp | 10/3 | 0.1846 |
| soft − FullHist | −22.2pp | −38.9pp--−5.6pp | 2/10 | 0.1250 |
| ArbiterKB − FullHist | −5.6pp | −22.2pp--11.1pp | 4/6 | 0.7539 |
| ArbiterKB − MEMI | −25.0pp | −41.7pp--−8.3pp | 2/11 | 0.1123 |
| ArbiterKB − soft | +16.7pp | 5.6pp--30.6pp | 6/0 | 0.1250 |
| MEMI − soft | +41.7pp | 25.0pp--58.3pp | 16/1 | **0.00165** |

没有任何单任务 contrast 在各自六项 Holm family 中显著。每个 task 只有 9 对，区间
较宽；这些结果不能外推为标准机器人总体成功率，也不能证明 MEMI 普遍优于所有方法。

## 轨迹、记忆与 watchdog 审计

144 个 event logs 共 4,039 条耐久事件。所有 36 个 task×episode 组都恰有四臂、
同一 episode seed 和同一 reset digest；36 个 episode 分别有 36 个独立 seed/digest。
577/577 次模型调用均 `valid`，返回模型全部为 `openai/gpt-6-luna`，0 retry、0 API
error；577 个 executor commands 均被日志捕获。共有 572 个完成的
`memory_transition`；5 个 timeout 在 watchdog 杀死时尚未完成最后一次 transition，
其部分事件与 token usage 仍由 supervisor 保存。

动作轨迹暴露了 soft repair 的系统性问题：15 个 soft 和 1 个 MEMI loop signature
被 diagnostic detector 记录，但按冻结规则没有提前终止。Button/soft 有 4 格达到
600s episode timeout；另 4 格执行满 20 个高层动作后失败。后四格原始 outcome 名称
是历史遗留的 `preflight_complete`，但其结构化 terminal status 一致为
`task_failure`，没有被误计成功。MEMI 另有 1 个 Button timeout。其余 139 格为
65 success 和 74 task failure。

只有 VideoUnmaskSwap 的 soft/Arbiter 日志产生可评分 structured beliefs，各 4 条，
样本过小，不能据此声称 belief calibration 优势。任务成功率也不能反推 belief quality。

## 成本与延迟

成本按冻结 manifest 单价从落盘 usage 计算。

| arm | calls | input tokens | output tokens | model latency | 总成本 | 平均成本/格 |
|---|---:|---:|---:|---:|---:|---:|
| FullHist | 88 | 206,857 | 6,799 | 227.635s | $0.013636 | $0.000379 |
| MEMI | 129 | 380,763 | 34,195 | 575.882s | $0.056368 | $0.001566 |
| soft | 246 | 436,329 | 27,481 | 734.941s | $0.038083 | $0.001058 |
| ArbiterKB | 114 | 268,924 | 12,742 | 379.655s | $0.025626 | $0.000712 |

Validation 合计 1,292,873 input、81,217 output、67,390 reasoning tokens，模型延迟
1,918.113s，估算成本 `$0.13371252`。串行事件窗口为 23,989.6s（6h39m50s），高于
预估 4.6h，但低于冻结 24h hard watchdog。Probe + smoke + validation 总估算费用
`$0.14223041`，低于 validation 的 `$0.50` stop budget。

与 FullHist 成对相比，ArbiterKB 每格多花 `$0.000333`
（95% bootstrap `$0.000188`--`$0.000556`）、多 4.223s 模型延迟
（0.328--9.664s）、多 0.72 次调用（0.08--1.47），同时成功率点估计更低。这直接
反对“当前完整 KB 以额外复杂度换来闭环收益”的主张。

## 泄漏与复现

- 成功 run 全部是 `split=val`、episodes 1--9；没有 test row。
- 没有启动 observation degradation，没有读取 held-out，没有新增 test ledger READ，
  没有根据 test 调参。
- 原始 288 个 `result.json/events.jsonl` 的路径+内容集合 SHA-256 为
  `1f9aba181041b319df7ae5e39e970794ab8150f85e8e681f50597ad37fe59cf5`。
- 机器可读 summary、完整 raw traces 与 reproducibility manifest 都在
  `results/robomme_common/watchdog_v2_1/validation_20261031T150046Z/`。
- 报告可由冻结 manifest 和 raw rows 重新生成；timeout 的 reset digest 从 supervisor
  的 durable reset event 恢复用于审计，不改写原始 result。
- 最终验证：278 个 pytest 全部通过；quality 134 files、0 problems；selftest
  16 PASS、0 FAIL、1 SKIPPED（缺少可选 real oracle replays）。

## 最终 Go/No-Go

**技术结论：GO。** common-harness validation 已可靠完成，可辩护地报告 144 格结果。

**研究结论：NO-GO。** 不读取 held-out，不做 degradation，不扩展 KB 规则。当前
ArbiterKB 既未超过 FullHist，也未超过 MEMI，且在 VideoRepick 出现 0/9 的结构性
失败；soft baseline 还显示高延迟、循环和 timeout。下一项工作若要继续，应先用现有
val traces 定位 VideoRepick 的失败机制并提出新的、预注册的研究假设；在没有新假设
前，保留 test split 未读比扩大实验更有研究价值。
