import os
import easydict


H36M_JOINTS = {
      0: 'Hip',          # Root joint (pelvis/center)
      1: 'RHip',         # Right hip
      2: 'RKnee',        # Right knee
      3: 'RAnkle',       # Right ankle
      4: 'LHip',         # Left hip
      5: 'LKnee',        # Left knee
      6: 'LAnkle',       # Left ankle
      7: 'Spine',        # Lower spine
      8: 'Thorax',       # Upper spine/chest
      9: 'Neck',         # Neck base
      10: 'Head',        # Head top
      11: 'LShoulder',   # Left shoulder
      12: 'LElbow',      # Left elbow
      13: 'LWrist',      # Left wrist
      14: 'RShoulder',   # Right shoulder
      15: 'RElbow',      # Right elbow
      16: 'RWrist'       # Right wrist
  }


H36M_JOINT_PAIRS = [
      # RIGHT LEG (3 bones)
      [0, 1], [1, 2], [2, 3],      # Hip → RHip → RKnee → RAnkle
      # LEFT LEG (3 bones)
      [0, 4], [4, 5], [5, 6],      # Hip → LHip → LKnee → LAnkle
      # SPINE & HEAD (4 bones)
      [0, 7], [7, 8], [8, 9],      # Hip → Spine → Thorax → Neck
      [9, 10],                      # Neck → Head
      # LEFT ARM (3 bones)
      [8, 11], [11, 12], [12, 13], # Thorax → LShoulder → LElbow → LWrist
      # RIGHT ARM (3 bones)
      [8, 14], [14, 15], [15, 16], # Thorax → RShoulder → RElbow → RWrist
  ]



VIT_JOINTS_NAME =('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist','L_Hip','R_Hip','L_Knee','R_Knee', 'L_Ankle','R_Ankle')
VIT_SKELETON= [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),]
VIT_SKELETON_COLOR = ['red','green','red','green','red','red','red','green','green','red','green','red','red','red','green','green']





PATH_ASSETS         = '../assets/'
PATH_DATASET        =  '/home/fylwen/WS/blade_project/assets/release_v1'
PATH_DETECTIONS = os.path.join(PATH_ASSETS, 'detections')
PATH_DETECTION_BBOXES = os.path.join(PATH_DETECTIONS, 'bbox')
PATH_DETECTION_VITPOSE = os.path.join(PATH_DETECTIONS, 'vitpose')
PATH_DETECTION_ALPHAPOSE = os.path.join(PATH_DETECTIONS, 'alphapose')
PATH_DETECTION_SAM2 = os.path.join(PATH_DETECTIONS, 'sam2')

INVALID_JOINTS_BY_SUBJECT = {
    'P1': ['RAnkle'],
    'P2': ['LAnkle', 'LKnee'],
    'P3': ['LAnkle'],
    'P5': ['RAnkle', 'RKnee'],
    'P7': ['RAnkle', 'LAnkle'],
    'P8': ['RAnkle', 'LAnkle'],}


def get_eval_model_free_cfg(args):
    cfg = easydict.EasyDict()
    cfg.evaluate_tracking = args.evaluate_tracking if hasattr(args, 'evaluate_tracking') else False
    cfg.use_gt_2d = args.use_gt2d if hasattr(args, 'use_gt2d') else False

    cfg.root_idx = 0  # Root joint index for PA-MPJPE and blade evaluation
    cfg.blade_grid_size = 8  # Grid size for sampling blade points from point_map
    cfg.blade_depth_outlier_threshold = 0.8  # Max depth difference (m) between blade points and root joint
    cfg.blade_depth_stability_patch = 5
    cfg.blade_depth_stability_thresh = 0.005
    cfg.image_shape = (2160, 3840)  # Original image shape (H, W)

    # Blade evaluation parameters
    cfg.blade_num_samples_along = 6  # Number of samples along blade rulings
    cfg.blade_num_samples_across = 6  # Number of samples across blade width
    cfg.blade_chamfer_threshold = [0.100, 0.050, 0.030]  # F1 thresholds: 100mm, 50mm, 30mm
    cfg.blade_chamfer_threshold_centered = [0.050, 0.030, 0.010]  # Centered F1 thresholds: 50mm, 30mm, 10mm


    return cfg