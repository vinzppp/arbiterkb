---
description: Run both independent reviews (code + goal) for a finished phase
argument-hint: <phase, e.g. M0>
---
Phase $ARGUMENTS claims to be done. Launch the `code-reviewer` and `goal-reviewer` agents in parallel, each with the request "Review phase $ARGUMENTS". Do not give them your own summary of the work; they read the repo themselves. When both reports exist in `reviews/`, print the two verdicts and every BLOCK / OFF-TRACK finding verbatim. Then stop and wait for the user. Do not start fixing anything.
