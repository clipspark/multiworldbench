"""Fixed isotropic pose scale from a disjoint, recorded calibration action."""
import numpy as np

def fit_scale(poses, reference_positions, indices, min_reference_distance=0.5, min_pose_distance=1e-6):
    indices=np.asarray(indices,dtype=int)
    if len(indices)<3 or np.any(np.diff(indices)!=1):
        raise ValueError('calibration requires at least three contiguous valid frames')
    p=np.asarray(poses)[indices,:3,3];q=np.asarray(reference_positions)[indices]
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('non-finite calibration input')
    estimated=float(np.linalg.norm(p[-1]-p[0]));reference=float(np.linalg.norm(q[-1]-q[0]))
    if reference<min_reference_distance or estimated<min_pose_distance:
        raise ValueError('calibration displacement is too small')
    return {'scale_multiplier':reference/estimated,'estimated_displacement':estimated,'reference_displacement':reference,'frames':indices.tolist(),'method':'heldout_endpoint_displacement_ratio_isotropic'}

def apply_scale(poses, multiplier):
    if not np.isfinite(multiplier) or multiplier<=0:raise ValueError('invalid scale')
    p=np.array(poses,dtype=float,copy=True)
    origin=p[0,:3,3].copy();p[:,:3,3]=origin+(p[:,:3,3]-origin)*multiplier
    return p

def fit_segments(poses, reference_positions, valid_index_segments):
    """Future capture prefix: fit one scalar over independently held-out motions."""
    segments=[fit_scale(poses,reference_positions,indices) for indices in valid_index_segments]
    if not segments:raise ValueError('no usable calibration motions')
    estimated=np.array([s['estimated_displacement'] for s in segments])
    reference=np.array([s['reference_displacement'] for s in segments])
    return {'scale_multiplier':float(estimated@reference/(estimated@estimated)),
            'method':'least_squares_displacement_through_origin','segments':segments}
