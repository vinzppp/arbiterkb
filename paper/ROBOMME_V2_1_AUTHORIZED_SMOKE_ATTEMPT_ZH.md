# RoboMME v2.1 授权 smoke 尝试报告

日期：2026-10-31（America/Los_Angeles）

分支：`research/perception-belief-robomme`；起点：`03cbdad`；`main` 未修改。

## 结论

冻结模型 `openai/gpt-6-luna` 仍出现在 OpenRouter 官方模型目录中，目录声明它支持
图像输入和 `/api/v1/chat/completions`。这只证明公开目录中的供应状态，不能证明当前
账户已获授权或余额、路由和实际返回模型符合协议。

授权端点验证与 16-cell smoke **没有启动**。当前进程环境、项目的 git-ignored
`.env`、launchd 环境和常用 macOS Keychain 服务均没有新的 OpenRouter credential。
此前暴露的 credential 没有被读取、恢复或使用。按 v2.1 fail-closed 规则，不能用
公开模型目录代替带授权的 endpoint probe，也不能静默换模型。

本轮没有创建新的 smoke run ID 或结果目录，没有调用付费 API，费用为 `$0.00`。
validation、observation degradation 和 held-out 均未启动或读取。

当前闸门：**NO-GO（authorized transport check 与 16-cell smoke）**。阻塞条件是运行
环境没有可用的新授权 credential，而不是模型从公开目录消失。只有安全环境向 runner
提供新的 `OPENAI_API_KEY`，并将 `OPENAI_API_URL` 精确设为冻结 endpoint 后，才能
执行授权模型确认并创建全新的 smoke run ID。模型或 endpoint 不匹配时仍必须停止，
不得修改冻结协议。

可审计记录：
`results/robomme_common/watchdog_v2_1/authorized_probe_20261031T1410Z.json`。
