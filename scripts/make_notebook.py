"""Writes notebooks/m0_walkthrough.ipynb and m1_walkthrough.ipynb (notebooks only call the package)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
M0 = [
    ("md", "# M0 walkthrough: one episode through ArbiterKB\nObservations → KB (beliefs + evidence) → plan → score. "
           "Synthetic RoboMME-like data, CPU only (DEVELOPMENT.md §6 M0). Each cell says what goes in and out."),
    ("code", "import sys, os\nsys.path.insert(0, '..')\nos.environ['ARBITER_DEBUG'] = '1'\n"
             "from arbiter_kb.sources.synthetic import SyntheticSource, COVER_T\n"
             "from arbiter_kb.sources.corrupt import CorruptedSource\n"
             "from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem\nfrom arbiter_kb.systems.baselines import TrackerSystem\n"
             "from arbiter_kb.views import text_view, ask\nfrom arbiter_kb.debug import explain\n"
             "from arbiter_kb.evaluate import run\nfrom arbiter_kb.params import Params\n"
             "params = Params.load('../params.json')  # the frozen parameter set (E5c)"),
    ("code", "# Perception stand-in: a Source yields one FrameObs per frame (visible / xyz in m / uv in px per object)\n"
             "src = SyntheticSource()\nep = 1025\nmeta, gt = src.meta(ep), src.ground_truth(ep)\n"
             "frames = list(src.frames(ep))\nprint(meta.task_goal, '| frames:', len(frames))\n"
             "print('truth:', gt.cube_to_bin, '| t*:', gt.t_star)\nframes[COVER_T].objects[gt.targets[0]]"),
    ("code", "# Fast loop + KB: step frame by frame; print the text view (what the arbiter will read)\n"
             "s = ArbiterKBSystem(params)\ns.reset(meta)\nshow = {COVER_T, 120, len(frames) - 1}\n"
             "for f in frames:\n    s.step(f)\n    if f.t in show:\n        print(f'--- t={f.t}')\n        print(text_view(s.kb))"),
    ("code", "# Evidence log: numbered so beliefs can cite it\nfor e in s.kb.evidence:\n    if e.kind in ('disappear', 'near_pass'):\n"
             "        print(e.id, e.t, e.kind, e.args, {k: v for k, v in e.data.items() if k != 'obs'})"),
    ("code", "# Predicate view with the course's unify / ASK: answers carry probabilities\ncube = gt.targets[0]\n"
             "ask(s.kb, ('Inside', cube, '?c'))"),
    ("code", "# Planner -> skill -> 4 waypoints [x, y, z, roll, pitch, yaw, gripper], positions from the KB's L0 only\n"
             "step = s.plan(cube)\nprint(step.skill, step.target, round(step.conf, 3), step.decision)\nstep.waypoints.round(3)"),
    ("code", "# Why does it believe that? The support chain down to raw observations (E2)\nprint(explain(s.trace_rows(), cube))"),
    ("md", "**Same episode with a tracker identity swap.** After the closest approach the two labels are exchanged. "
           "Bound follows the wrong label with confidence 1. ArbiterKB moved only a little mass to the other "
           "container, so it is ALSO wrong with confidence above τ_hi = 0.8: a silent failure. Since M1, R6 uses the "
           "3D distance of the pass, so at lane 0.07 it moves almost no mass. (This corruption teleports the label "
           "~12 cm in one frame, which no real tracker does: sanity only; see NOTES and m1_walkthrough.)"),
    ("code", "bad = CorruptedSource(src, p=1.0)\n"
             "rows, traces, _ = run([TrackerSystem('bound', params), ArbiterKBSystem(params)], bad, [ep], params, 'id_swap')\n"
             "print(bad.log[ep])\n[(r['system'], r['pred'], r['truth'], r['p_max'], r['correct']) for r in rows]"),
    ("code", "# Summary figure from the full run (make selftest writes it)\nfrom IPython.display import Image\n"
             "Image('../figures/m0_belief_timeline_synthetic_p1.0.png')"),
]

M1 = [
    ("md", "# M1 walkthrough: realistic-looking id swaps and three cheap baselines\nPlumbing only: default "
           "parameters, nothing tuned (DEVELOPMENT.md §6 M1). The corruption is ours, so the question here is "
           "whether everything runs and explains itself, not whether ArbiterKB wins."),
    ("code", "import sys, os\nsys.path.insert(0, '..')\nos.environ['ARBITER_DEBUG'] = '1'\nimport numpy as np\n"
             "from arbiter_kb.sources.synthetic import SyntheticSource\n"
             "from arbiter_kb.sources.realistic import OverlapSwapSource, NoisySource\n"
             "from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem\n"
             "from arbiter_kb.systems.baselines import TrackerSystem, GuardedBound\n"
             "from arbiter_kb.evaluate import run\nfrom arbiter_kb.debug import explain\nfrom arbiter_kb.params import Params\n"
             "params = Params.load('../params.json')"),
    ("code", "# lane 'mix': each episode draws a sideways offset; small offsets make the two containers overlap\n"
             "base = SyntheticSource(lane_offset='mix', swap_steps='mix')\n"
             "[(ep, base.settings(ep)) for ep in base.episodes('train', 8)]"),
    ("code", "# id_swap_overlap: ids may swap only where the two tracks come within d_ov = 3 cm; the label jumps < d_ov\n"
             "bad = OverlapSwapSource(base, p_swap=1.0, d_ov=params.baseline.d_ov)\n"
             "ep = next(e for e in base.episodes('train', 40) if list(bad.frames(e)) and bad.log[e]['corrupted'])\n"
             "base.settings(ep), bad.log[ep]"),
    ("code", "# Six systems on that episode; correctness is physical (the label nearest the true container)\n"
             "systems = [TrackerSystem('frozen', params), TrackerSystem('bound', params), GuardedBound('haircut', params),\n"
             "           GuardedBound('overlap_haircut', params), GuardedBound('jump', params), ArbiterKBSystem(params)]\n"
             "rows, traces, _ = run(systems, bad, [ep], params, 'id_swap_overlap')\n"
             "[(r['system'], r['pred'], r['truth'], r['p_max'], r['correct']) for r in rows]"),
    ("code", "# R6 explained: the near_pass evidence with min_3d, and the swap mixing weight it produced\n"
             "cube = base.ground_truth(ep).targets[0]\nprint(explain(traces[('arbiterkb', ep)], cube))"),
    ("code", "# Baselines explain too: one-hot answer plus the doubt trigger\n"
             "print(explain(traces[('overlap_haircut', ep)], cube))"),
    ("md", "**Reading the table.** OverlapHaircut is Bayes-optimal under id_swap_overlap because we wrote both: "
           "ArbiterKB − OverlapHaircut ≈ 0 here is expected and says nothing about real trackers. That is why "
           "phase G measures a real tracker's swaps before M2 tunes anything. Read silent failure together with "
           "coverage (share answered with p_max ≥ 0.8): a 50/50 haircut scores 0 silent failures by never being sure. "
           "AURC summarises the whole risk–coverage curve. JumpDetector equals Bound here (jumps < j_max) and "
           "collapses under position noise."),
    ("code", "import pandas as pd, glob\npd.read_csv('../results/latest/m1_1_table/summary.csv')"
             "[['condition', 'system', 'acc', 'silent', 'coverage', 'aurc', 'brier', 'silent_on_corrupted']]"),
    ("code", "from IPython.display import Image, display\nfor f in sorted(glob.glob('../figures/*_test_p0.5.png')):\n"
             "    print(f); display(Image(f))"),
]
NOTEBOOKS = {"m0_walkthrough": M0, "m1_walkthrough": M1}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        out = ROOT / "notebooks" / f"{name}.ipynb"
        out.parent.mkdir(exist_ok=True)
        nbf.write(nb, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
