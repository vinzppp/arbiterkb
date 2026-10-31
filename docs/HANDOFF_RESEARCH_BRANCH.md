# Handoff: `research/perception-belief-robomme`

## State

The requested pre-existing branch/R1c ref was not present locally. This branch was reconstructed from the latest
auditable source commit `069beb7`; the original source repository and `main` were not modified.

Implemented and run:

- `arbiter_kb/paper_stats.py`: Wilson intervals, exact McNemar, episode-paired bootstrap;
- `scripts/paper_compare.py`: frozen held-out perceived repair ladder plus direct parsing and verification of MEMI's
  released closed-loop logs;
- `results/latest/paper_comparison/summary.json`: complete machine-readable output and data hashes;
- `paper/PERCEPTION_BELIEF_COMPARATIVE_REPORT.md`: paper-style results and conclusions;
- `paper/DIRECT_CLOSED_LOOP_PROTOCOL.md`: frozen protocol for the unrun common-harness test.

MEMI was cloned temporarily at commit `bb6c37c`; `python -m memi.report` ended with `ALL MATCH`. No API was called.
The local RoboMME simulator preflight cannot start (`robomme` absent; macOS/no NVIDIA), recorded in
`results/latest/paper_comparison/local_environment_audit.json`.

## Result in one sentence

MEMI's released closed-loop gains reproduce, but on ArbiterKB's held-out perceived identity benchmark the full KB does
not beat causal re-linking plus detector-only soft repair on accuracy or AURC; a direct closed-loop comparison remains
unrun and must not be claimed.

## Reproduce

```bash
make check
.venv/bin/python scripts/paper_compare.py --memi-root /path/to/memi --split test
```

The test read is append-only in `results/test_reads.log`. Re-running is an audit repeat, not a new independent test.
