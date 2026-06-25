import numpy as np
import torch
import math
import cv2
import os
from torch.nn import functional as F

from utils.camera import project_3d_with_intrinsics
from utils.process_utils import upsample_ruled_surface, generate_silhouette_mask, compute_convex_hull_2d
from utils.chamfer_distance import compute_fscore

def eval_3d_pose(pred, target, root_idx=None, eval_joints=None, valid_joints=None, verbose=False):
    pred, target = pred.copy(), target.copy()#[bs, n_joints,3]
    batch_size = pred.shape[0]

    if verbose:
        print("pred",pred.shape, "target",target.shape)
        print("root_idx",root_idx, "eval_joints",eval_joints)#, "valid_joints",valid_joints.shape)

    if type(root_idx) is tuple:
        assert False
        pred_root = (pred[:, root_idx[0], :]+pred[:, root_idx[1], :])/2
        target_root = (target[:, root_idx[0], :]+target[:, root_idx[1], :])/2
    else:
        pred_root = pred[:, root_idx, :].copy()
        target_root = target[:, root_idx, :].copy()
    
    if verbose:
        print("pred_root",pred_root.shape, "target_root",target_root.shape)
    pred = pred - pred_root[:, None, :]
    target = target - target_root[:,None,:]

    pred, target = pred[:, eval_joints, :], target[:, eval_joints, :]

    if valid_joints is not None:
        valid_joints = valid_joints[:, eval_joints]
    
        
    mpjpe, pa_mpjpe = [], []
    preds_in_pa = []
    for j in range(batch_size):
        cpred=pred[j]
        ctarget=target[j]
        if valid_joints is not None:
            cvalid = valid_joints[j]
            cpred = cpred[cvalid>0]
            ctarget = ctarget[cvalid>0]
        
        if ctarget.shape[0]<5:
            print(f"Warning: not enough valid joints for sample {j}, skipping MPJPE/PA-MPJPE evaluation for this sample.")
            mpjpe.append(-1)
            pa_mpjpe.append(-1)
            continue
            
        mpjpe.append(eval_mpjpe(cpred, ctarget))
        cpa_mpjpe, cpred_in_pa = eval_pa_mpjpe(cpred, ctarget)
        pa_mpjpe.append(cpa_mpjpe)
        preds_in_pa.append(cpred_in_pa)
    
    if verbose:
        to_vis=np.concatenate([target,np.array(preds_in_pa)],axis=1)
        to_vis = np.tile(to_vis, (5, 1, 1))
        render_animation(to_vis, kinematics,'hello_pa.gif',colors=['red','blue','red','red','blue'],input_zup=False, has_gt=True, figsize=(5,5))
        print(mpjpe, pa_mpjpe)
        exit(0)
    return mpjpe, pa_mpjpe


def eval_mpjpe(predicted, target):
    return np.mean(np.sqrt(np.sum((predicted - target) ** 2, 1)))

def eval_pa_mpjpe(predicted, target):        
    predicted = rigid_align(predicted, target)
    return eval_mpjpe(predicted, target), predicted

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2



def eval_mesh(pred, target, pred_joint_cam, gt_joint_cam, root_idx):
    pred, target = pred.copy(), target.copy()
    batch_size = pred.shape[0]
    
    pred, target = pred - pred_joint_cam[:, None, root_idx, :], target - gt_joint_cam[:, None, root_idx, :]
    
    mpvpe = []
    for j in range(batch_size):
        mpvpe.append(eval_mpjpe(pred[j], target[j]))
    
    return mpvpe

def eval_accel_error(joints_pred, joints_gt, root_idx, eval_joints, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    joints_pred, joints_gt = joints_pred.copy(), joints_gt.copy()    
    joints_pred, joints_gt = joints_pred - joints_pred[:, None, root_idx, :], joints_gt - joints_gt[:, None, root_idx, :]
    joints_pred, joints_gt = joints_pred[:, eval_joints, :], joints_gt[:, eval_joints, :]

    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)



#code adopt from VIBE
def batch_compute_similarity_transform_torch(S1, S2, transposed=True, return_transformation=False, verbose=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    if transposed:  #if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
    assert(S2.shape[1] == S1.shape[1])#to [B, 3, N]?

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    if verbose:
        print("S1",S1.shape, "S2",S2.shape)#should be [B, 3, N]
        print("mu1",mu1.shape, "mu2",mu2.shape)#should be [B, 3, 1]

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    if verbose:
        print("var1",var1.shape)#should be [B]
        print("K",K.shape)#should be [B, 3, 3]
        print("U",U.shape, "s",s.shape, "V",V.shape)#should be [B, 3, 3], [B, 3], [B, 3, 3]

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)
    
    if return_transformation:
        return S1_hat, {'scale': scale, 'rotation': R, 'translation': t}

    return S1_hat




def eval_batch_seq_pose3d_torch(batch_seq_pred, batch_seq_target, verbose=False):
    batch_size, len_seq = batch_seq_pred.shape[0:2]

    batch_seq_mpjpe=torch.mean(torch.norm(batch_seq_pred-batch_seq_target, dim=-1), dim=-1)
    seq_mpjpe=torch.sum(batch_seq_mpjpe, dim=0)

    #remove root by following CycleAdapt
    batch_seq_pred_ra = (batch_seq_pred - batch_seq_pred[:, :, 0:1, :])[:,:,1:]
    batch_seq_target_ra = (batch_seq_target-batch_seq_target[:, :, 0:1,:])[:,:,1:]

    batch_seq_mpjpe_ra=torch.mean(torch.norm(batch_seq_pred_ra-batch_seq_target_ra, dim=-1), dim=-1)
    seq_mpjpe_ra=torch.sum(batch_seq_mpjpe_ra, dim=0)

    #compute procrustes alignment
    flatten_pred_ra=batch_seq_pred_ra.reshape(batch_size*len_seq, -1, 3)
    flatten_target_ra=batch_seq_target_ra.reshape(batch_size*len_seq, -1, 3)
    flatten_pred_pa=batch_compute_similarity_transform_torch(flatten_pred_ra, flatten_target_ra)
    
    batch_seq_pred_pa = flatten_pred_pa.reshape(batch_size, len_seq, -1, 3)
    batch_seq_mpjpe_pa=torch.mean(torch.norm(batch_seq_pred_pa-batch_seq_target_ra, dim=-1), dim=-1)
    seq_mpjpe_pa=torch.sum(batch_seq_mpjpe_pa, dim=0)


    if verbose:
        print("batch_seq_mpjpe",batch_seq_mpjpe.shape, seq_mpjpe.shape)# [bs,T]
        print("batch_seq_mpjpe_ra",batch_seq_mpjpe_ra.shape, seq_mpjpe_ra.shape)#[bs,T]
        print("batch_seq_mpjpe_pa",batch_seq_mpjpe_pa.shape, seq_mpjpe_pa.shape)#[bs,T]
        
        batch_seq_pred_ra=batch_seq_pred - batch_seq_pred[:, :, 0:1, :]
        batch_seq_target_ra=batch_seq_target-batch_seq_target[:, :, 0:1,:]
        for i in range(0, batch_size):
            cra,cpa=eval_3d_pose(batch_seq_pred_ra[i].cpu().numpy(), batch_seq_target_ra[i].cpu().numpy())
            assert np.fabs(np.array(cra)-batch_seq_mpjpe_ra[i].cpu().numpy()).max()<1e-6
            assert np.fabs(np.array(cpa)-batch_seq_mpjpe_pa[i].cpu().numpy()).max()<1e-6

    return_dict = {"batch_size": batch_size, "seq_mpjpe": seq_mpjpe, "seq_mpjpe_ra": seq_mpjpe_ra, "seq_mpjpe_pa": seq_mpjpe_pa, 
                    "batch_seq_mpjpe": batch_seq_mpjpe, "batch_seq_mpjpe_ra": batch_seq_mpjpe_ra, "batch_seq_mpjpe_pa": batch_seq_mpjpe_pa}
    return return_dict


def mask_area_recall(blade_3d_gt, blade_3d_est, image_shape, intr):
    blade_2d_gt = project_3d_with_intrinsics(blade_3d_gt.reshape(-1, 3), intr)  # (N*2, 2)
    blade_2d_gt = blade_2d_gt.reshape(-1, 2, 2)  # (N, 2, 2)
    blade_mask_gt, bbox_xyxy = generate_silhouette_mask(image_shape, blade_2d_gt)

    blade_2d_est = project_3d_with_intrinsics(blade_3d_est, intr)  # (M, 2)

    # Compute convex hull from blade_2d_est
    _, blade_mask_est, _ = compute_convex_hull_2d(points_2d=blade_2d_est, image_shape=image_shape)

    # Compute mask recall rate
    intersection = np.logical_and(blade_mask_est > 0, blade_mask_gt > 0).sum()
    gt_area = (blade_mask_gt > 0).sum()
    recall_rate = intersection / gt_area if gt_area > 0 else 0.0
    return recall_rate
    


def eval_blade(blade_3d_gt, blade_valid_gt, blade_3d_est,
                skeleton_3d_gt, skeleton_3d_est, root_idx=0,
                num_samples_along=25, num_samples_across=8,
                thre_fscore=0.03, thre_fscore_centered=0.01):

    results = {}

    # Helper: generate result key names for F1 scores.
    # List thresholds → per-threshold keys like 'f1_score_100', 'f1_score_centered_50'.
    # Scalar threshold → backward-compat keys 'f1_score', 'f1_score_centered'.
    def _f1_keys(thres, centered):
        prefix = 'f1_score_centered' if centered else 'f1_score'
        if isinstance(thres, (list, tuple)):
            return [f'{prefix}_{int(round(t * 1000))}' for t in thres]
        return [prefix]

    f1_keys  = _f1_keys(thre_fscore, False)
    f1c_keys = _f1_keys(thre_fscore_centered, True)

    blade_3d_gt_upsampled, blade_valid_upsampled = upsample_ruled_surface(
                                                        blade_3d=blade_3d_gt,
                                                        blade_valid=blade_valid_gt,
                                                        num_samples_along=num_samples_along,
                                                        num_samples_across=num_samples_across)

    # Get root positions
    root_gt = skeleton_3d_gt[root_idx].copy()
    blade_3d_gt_upsampled_aligned = blade_3d_gt_upsampled - root_gt

    # Filter valid points for evaluation
    blade_3d_gt_valid = blade_3d_gt_upsampled_aligned[blade_valid_upsampled]


    if blade_3d_gt_valid is None or len(blade_3d_gt_valid) == 0:
        results['chamfer_distance'] = -1
        results['chamfer_distance_centered'] = -1
        for k in f1_keys + f1c_keys:
            results[k] = -1
        return results

    # Blade not detected: return sentinel values without computing chamfer
    if blade_3d_est is None or len(blade_3d_est) == 0:
        results['chamfer_distance'] = -1
        results['chamfer_distance_centered'] = -1
        for k in f1_keys + f1c_keys:
            results[k] = 0
        return results

    root_est = skeleton_3d_est[root_idx].copy()
    blade_3d_est_aligned = blade_3d_est - root_est

    # Compute F1 scores (supports scalar or list thresholds via compute_fscore)
    fscores, chamfer_dist = compute_fscore(blade_3d_gt_valid, blade_3d_est_aligned, thres=thre_fscore)
    results['chamfer_distance'] = chamfer_dist
    if isinstance(thre_fscore, (list, tuple)):
        for k, fs in zip(f1_keys, fscores):
            results[k] = fs
    else:
        results[f1_keys[0]] = fscores

    # Further normalize: center both clouds then compute centered F1
    cent_gt = np.mean(blade_3d_gt_valid, axis=0)
    blade_3d_gt_centered = blade_3d_gt_valid - cent_gt

    cent_est = np.mean(blade_3d_est_aligned, axis=0)
    blade_3d_est_centered = blade_3d_est_aligned - cent_est

    fscores_c, chamfer_dist_c = compute_fscore(blade_3d_gt_centered, blade_3d_est_centered, thres=thre_fscore_centered)
    results['chamfer_distance_centered'] = chamfer_dist_c
    if isinstance(thre_fscore_centered, (list, tuple)):
        for k, fs in zip(f1c_keys, fscores_c):
            results[k] = fs
    else:
        results[f1c_keys[0]] = fscores_c


        
    return results



def eval_mean_valid(values):
    """Mean excluding -1 sentinel values (blade not detected)."""
    arr = np.array(values)
    valid = arr[arr >= 0]
    return float(np.mean(valid)) if len(valid) > 0 else float('nan')