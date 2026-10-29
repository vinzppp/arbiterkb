# Archive

Kept for history and provenance; nothing here is used by the code, `make check` or the runbooks. NOTES.md still cites
some of these paths by their old names: a `results/latest/<name>` that is not in `results/latest/` is in
`archive/results/<name>`.

| Path | What it was | Superseded by |
|---|---|---|
| `docs/INSTRUCTIONS_v1.0.md` | the first build instructions | `DEVELOPMENT.md` (spec v1.1 onwards) |
| `docs/M0_walking_skeleton_prompt.md` | the M0 build prompt | `DEVELOPMENT.md` §6 M0 |
| `notebooks/`, `executed/` | M0 / M1 walkthrough notebooks and `make_notebook.py` (was `make notebook`) | NOTES.md and `results/latest/` |
| `results/count_robot_state*` (v1, v2, counter v1 / v2, words, words_trim) | counting tests before v3 | `results/latest/count_robot_state_v3`, `count_robot_state_counter_v3`, `count_controls` |
| `results/describe_*`, `results/review_video*` | the periodic VLM review call and window descriptions | removed in spec v1.11 (`ask.py`) |
| `results/keyframe_check` | the v1.9 merged keyframe prompt | `results/latest/keyframe_check_v110` |
| `results/goal_parse` | goal parse v1 / v2 | `results/latest/goal_parse_v3` |
| `results/m1_table`, `m1_1_table`, `synthetic_test_*` | M0 / M1 synthetic tables | `results/latest/m2_infra_*`, `claim3_g` |
| `results/m4_dry_text` | the first text-only dry run | `results/latest/m4_batchA` |
| `uploads/` (git-ignored) | old pod uploads: `pod_code.tar.gz` (G), `g_perceived.tar.gz` (G's outputs; unpacked in `data/g_perceived/`) | `pod_upload_*.tar.gz` at the repo root |
