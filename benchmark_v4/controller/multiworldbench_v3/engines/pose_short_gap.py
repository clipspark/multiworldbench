"""Fill bounded short pose gaps inside individual action windows only."""
import numpy as np
from .rotation import interpolate_rotations

def fill_action_gaps(poses, valid, windows, fps=20, max_gap_seconds=0.15):
    output=np.array(poses, dtype=float, copy=True)
    observed=np.asarray(valid, dtype=bool) & np.isfinite(output).all(axis=(1,2))
    filled=np.zeros(len(output), dtype=bool)
    limit=int(np.floor(fps*max_gap_seconds+1e-9))
    for lo,hi in windows:
        lo,hi=max(0,int(lo)),min(len(output),int(hi))
        anchors=np.flatnonzero(observed[lo:hi])+lo
        for left,right in zip(anchors[:-1],anchors[1:]):
            missing=right-left-1
            if missing<1 or missing>limit:continue
            indices=np.arange(left+1,right)
            fraction=(indices-left)/(right-left)
            output[indices]=np.eye(4)
            output[indices,:3,3]=(1-fraction[:,None])*output[left,:3,3]+fraction[:,None]*output[right,:3,3]
            output[indices,:3,:3]=interpolate_rotations(np.array([left,right]),output[[left,right],:3,:3],indices)
            filled[indices]=True
    return output,observed|filled,filled
