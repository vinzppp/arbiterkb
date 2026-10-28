"""Writes notebooks/m0_walkthrough.ipynb (the notebook only calls the package; logic stays in arbiter_kb)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    ("md", "# M0 walkthrough: one episode through ArbiterKB\nObservations → KB (beliefs + evidence) → plan → score. "
           "Synthetic RoboMME-like data, CPU only (DEVELOPMENT.md §6 M0)."),
    ("code", "import sys, os\nsys.path.insert(0, '..')\nos.environ['ARBITER_DEBUG'] = '1'\n"
             "from arbiter_kb.sources.synthetic import SyntheticSource, COVER_T\n"
             "from arbiter_kb.sources.corrupt import CorruptedSource\n"
             "from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem\n"
             "from arbiter_kb.views import text_view, ask\nfrom arbiter_kb.debug import explain"),
    ("md", "**Perception stand-in.** A Source yields one `FrameObs` per frame: per object visible / xyz (m) / uv (px)."),
    ("code", "src = SyntheticSource()\nep = 1025\nmeta, gt = src.meta(ep), src.ground_truth(ep)\n"
             "frames = list(src.frames(ep))\nprint(meta.task_goal, '| frames:', len(frames))\n"
             "print('truth:', gt.cube_to_bin, '| t*:', gt.t_star)\nframes[COVER_T].objects[gt.targets[0]]"),
    ("md", "**Fast loop + KB.** Step frame by frame; print the KB's text view (what the arbiter will read) "
           "just after the cover, mid-run and at t*."),
    ("code", "s = ArbiterKBSystem()\ns.reset(meta)\nshow = {COVER_T, 120, len(frames) - 1}\n"
             "for f in frames:\n    s.step(f)\n    if f.t in show:\n        print(f'--- t={f.t}')\n        print(text_view(s.kb))"),
    ("md", "**Evidence log** (appear / disappear / move / near_pass), numbered so beliefs can cite it."),
    ("code", "for e in s.kb.evidence:\n    if e.kind in ('disappear', 'near_pass'):\n        print(e.id, e.t, e.kind, e.args, "
             "{k: v for k, v in e.data.items() if k != 'obs'})"),
    ("md", "**Predicate view** with the course's unify / ASK — answers carry probabilities."),
    ("code", "cube = gt.targets[0]\nask(s.kb, ('Inside', cube, '?c'))"),
    ("md", "**Planner → skill → 4 waypoints** `[x, y, z, roll, pitch, yaw, gripper]` from the KB's L0 only."),
    ("code", "step = s.plan(cube)\nprint(step.skill, step.target, round(step.conf, 3), step.decision)\nstep.waypoints.round(3)"),
    ("md", "**Why does it believe that?** The support chain down to raw observations (E2)."),
    ("code", "print(explain(s.trace_rows(), cube))"),
    ("md", "**Same episode with a tracker identity swap.** Labels lie after the closest approach; the bound rule "
           "follows the wrong label with confidence 1, ArbiterKB keeps the alternative."),
    ("code", "bad = CorruptedSource(src, p=1.0)\nfrom arbiter_kb.evaluate import run\nfrom arbiter_kb.params import Params\n"
             "from arbiter_kb.systems.baselines import TrackerSystem\n"
             "rows, traces, _ = run([TrackerSystem('bound'), ArbiterKBSystem()], bad, [ep], Params(), 'id_swap')\n"
             "print(bad.log[ep])\n[(r['system'], r['pred'], r['truth'], r['p_max'], r['correct']) for r in rows]"),
    ("md", "**Summary figure** from the full M0 run (`make selftest` writes it)."),
    ("code", "from IPython.display import Image\nImage('../figures/m0_belief_timeline.png')"),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / "m0_walkthrough.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
