"""Repeated solo-control trajectory consistency, adapted from WBench cnATE.

Standalone implementation: no WBench runtime dependency. This is a reliability
gate for isolation, not action correctness: two identical wrong trajectories can
agree. Threshold 0.60 is a provisional MWB policy, not a WBench threshold.
"""
import numpy as np
from ..engines.rotation import rotation_magnitude
from ..engines.navigation_core import rotation_steps

DEFAULT_THRESHOLD = 0.60
SCORING_VERSION = "solo_consistency.wbench_cnate.v1"

def score_case(evidence):
    threshold = float(evidence.get("threshold", DEFAULT_THRESHOLD))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite in [0,1]")
    refs = evidence.get("reference_poses")
    if refs is None or len(refs) != 2:
        return {"score": None, "eligible": False, "status": "not_evaluable",
                "reason": "two repeated solo reference trajectories required"}
    local = []
    for poses in refs:
        p = np.asarray(poses, dtype=float)
        if p.ndim != 3 or p.shape[1:] != (4,4) or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("references require at least 3 finite c2w poses")
        r0 = p[0,:3,:3].T
        local.append(((p[:,:3,3]-p[0,:3,3]) @ r0.T,
                      np.einsum("ij,njk->nik", r0, p[:,:3,:3])))
    n = min(len(x[0]) for x in local)
    local = [(p[np.linspace(0,len(p)-1,n).astype(int)],
              r[np.linspace(0,len(r)-1,n).astype(int)]) for p,r in local]
    (p,r),(q,s) = local
    path = (np.linalg.norm(np.diff(p,axis=0),axis=1).sum()+
            np.linalg.norm(np.diff(q,axis=0),axis=1).sum())/2
    rot = (rotation_steps(r).sum()+rotation_steps(s).sum())/2
    floors = evidence.get("normalization_floors", {"translation":0.5,"rotation":10.})
    ft,fr = float(floors["translation"]),float(floors["rotation"])
    if not np.isfinite([ft,fr]).all() or min(ft,fr)<=0: raise ValueError("positive finite floors required")
    ate_t = float(np.linalg.norm(p-q,axis=1).mean())
    ate_r = float(np.rad2deg(rotation_magnitude(np.einsum("nji,njk->nik",r,s))).mean())
    ct,cr = min(ate_t/max(path,ft),1.),min(ate_r/max(rot,fr),1.)
    score = 1-(ct+cr)/2
    return {"score":float(score),"eligible":bool(score>=threshold),
            "status":"numeric","threshold":threshold,"cnATE_t":ct,"cnATE_r":cr,
            "ATE_t":ate_t,"ATE_r_deg":ate_r,"mean_reference_path":float(path),
            "mean_reference_rotation_deg":float(rot),"sample_count":n,
            "scoring_version":SCORING_VERSION,"threshold_status":"provisional_mwb_policy"}

def aggregate_cases(rows):
    vals=[r["score"] for r in rows if r.get("score") is not None]
    return {"score":float(np.mean(vals)) if vals else None,"evaluated":len(vals),
            "eligible":sum(bool(r.get("eligible")) for r in rows),"total":len(rows)}
