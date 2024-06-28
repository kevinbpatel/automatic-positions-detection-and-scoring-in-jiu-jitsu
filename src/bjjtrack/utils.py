import os
import shutil

import numpy as np


def get_bbox_from_pose(pose, pad = 0.10):
    xmin = pose[:, 0].min()
    xmax = pose[:, 0].max()
    w = xmax - xmin
    padx = w * pad
    ymin = pose[:, 1].min()
    ymax = pose[:, 1].max()
    h = ymax - ymin
    pady = h * pad
    conf_max = pose[:, 2].max()
    xmin = xmin - padx 
    xmax = xmax + padx
    ymin = ymin - pady
    ymax = ymax + pady
    bbox = np.array([xmin, ymin, xmax, ymax, conf_max])
    return {'bbox' : bbox}


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    pred_dir = os.path.join(prefix, "predictions")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    if os.path.exists(pred_dir) and os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    return pose_dir, pred_dir


#Adapted from: 
# https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/apis/inference_tracking.py
def vis_pose_tracking_result(model,
                             img,
                             result,
                             radius=4,
                             thickness=1,
                             kpt_score_thr=0.3,
                             dataset='TopDownCocoDataset',
                             dataset_info=None,
                             show=False,
                             out_file=None,
                             sort = False,
                             vis_bg = False
                             ):
    """Visualize the pose tracking results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    # sort the results by track_id
    if sort and vis_bg:
        result = sorted(result, key=lambda x: x['track_id'], reverse=True)

    palette = np.array([[200,   0, 0], [0,   0, 200], 
                        [  0, 255, 0], [0, 255, 255]])

    if dataset_info is None and dataset is not None:
        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset'):
            kpt_num = 17
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]
        else:
            raise NotImplementedError()

    elif dataset_info is not None:
        kpt_num = dataset_info.keypoint_num
        skeleton = dataset_info.skeleton

    for res in result:
        track_id = res['track_id']
        if not vis_bg and track_id > 1:
            # tracking idis > 1 are background (TODO: 2 might be referee in the future)
            continue
        bbox_color = palette[track_id % len(palette)]
        pose_kpt_color = palette[[track_id % len(palette)] * kpt_num]
        pose_link_color = palette[[track_id % len(palette)] * len(skeleton)]
        img = model.show_result(
            img, [res],
            skeleton,
            radius=radius,
            thickness=thickness,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            bbox_color=tuple(bbox_color.tolist()),
            kpt_score_thr=kpt_score_thr,
            show=show,
            out_file=out_file)

    return img