# Adapted from https://github.com/ViTAE-Transformer/ViTPose/blob/main/demo/top_down_pose_tracking_demo_with_mmdet.py

import os
import copy
import time
import pickle
import logging
import warnings
from argparse import ArgumentParser

import cv2
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector


from bjjtrack.tracking import Tracker, JOINTS, add_featuremaps
from bjjtrack.utils import get_bbox_from_pose, prepare_output_dirs, vis_pose_tracking_result

warnings.filterwarnings("ignore")

DIST_TRESH = 0.036
POSE_WINDOW_LEN = 10
DATASET_UPDATE_FREQ = 14
CLASSIFIER_UPDATE_FREQ = 140
JOINT_CONF_TRESH = 0.62
NJOINTS = 14
VIS_CONF_TRESH = 7


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tracker-path', type=str, default='checkpoints/tracker/tracker_v3.pickle')
    parser.add_argument('--det-config', help='Config file for detection', default='mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py')
    parser.add_argument('--det-checkpoint', help='Checkpoint file for detection', default='checkpoints/detection/deformable_detr_twostage_refine.pth')
    parser.add_argument('--pose-config', help='Config file for pose', default='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
    parser.add_argument('--pose-checkpoint', help='Checkpoint file for pose', default='checkpoints/vitpose-h-multi-coco.pth')
    parser.add_argument('--init-frames-count', type=int, default=150, help='Number of frames to initialize the tracker using only pose distances. (5s by default)')
    
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--skip-frames-count', type=int, default = 0)
    parser.add_argument('--out-root', default='outputs/demo', help='Root of the output files.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--device-det', default='cuda:0', help='Device used for detection')
    parser.add_argument('--bbox-thr', type=float, default=0.1, help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=2, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--no-vis-bg', action='store_true', help='Do not visualize the unmatched poses.')

    return parser.parse_args()


def main():
    """
    Run tracking pipeline from https://dl.acm.org/doi/10.1145/3552437.3555707.
    """
    args = parse_args()
    pose_dir, pred_dir = prepare_output_dirs(prefix=args.out_root)
    logfile = os.path.join(args.out_root, "log")
    open(logfile, 'w').close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("tracking")

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    init_frames_count = args.init_frames_count
    tracker_path = args.tracker_path
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device_det.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    fps = None

    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        os.path.join(args.out_root,
                        f'vis_{os.path.basename(args.video_path)}'), fourcc,
        fps, size)

    frame = 0
    tracker = Tracker(tracker_path, 2, size, 
                      POSE_WINDOW_LEN, DIST_TRESH, 
                      JOINTS, NJOINTS, JOINT_CONF_TRESH, 
                      DATASET_UPDATE_FREQ, CLASSIFIER_UPDATE_FREQ, VIS_CONF_TRESH)
    
    pose1 = None
    pose2 = None
    predictions = dict()
    while (cap.isOpened()):
        flag, img = cap.read()
        frame += 1
        
        if not flag:
            break
        if frame < args.skip_frames_count:
            continue

        then = time.time()
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, 1)

        #improve detections by adding the extended bounding boxes of previous poses
        if pose1:
            person_results.append(get_bbox_from_pose(pose1['keypoints']))
        if pose2:
            person_results.append(get_bbox_from_pose(pose2['keypoints']))

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=False)

        now = time.time()
        logger.info(f'Detected {len(pose_results)} persons pose in {now - then :.04}s Frame {frame}')
        
            
        if len(pose_results):
            pose_results = add_featuremaps(pose_model, img, pose_results, args.device)
            if init_frames_count > frame - args.skip_frames_count:
                matched, pose_results, updated = tracker.init_tracking(pose_results, frame)
            else:
                matched, pose_results, updated = tracker.track(pose_results, frame)
            NO_MATCH = False
            if 0 in matched:
                id1 = matched[0]
                pose1 = pose_results[id1]
            else:
                pose1 = None
            if 1 in matched:
                id2 = matched[1]
                pose2 = pose_results[id2]
            else: 
                pose2 = None
        else:
            logger.info('No pose predictions')
            break

        predictions[frame] = copy.deepcopy(pose_results)


        if pose1 is None and pose2 is None:
            NO_MATCH = True
            logger.info('No matches')

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=0,
            show=False,
            sort = True,
            vis_bg = not args.no_vis_bg)


        if updated:
            cv2.imwrite(f"{pose_dir}/features{frame :05d}.jpg", vis_img)
        elif frame < args.skip_frames_count + 30 or frame % 100 == 0 or NO_MATCH:
            cv2.imwrite(f"{pose_dir}/{frame :05d}.jpg", vis_img)
          
        videoWriter.write(vis_img)


    
    print('Saving predictions', end='...')
    with open(f'{pred_dir}/predictions.pickle', 'wb') as f:
        pickle.dump(predictions, f)

    print('Done')
    tracker.save_data(args.out_root)


    cap.release()
    videoWriter.release()


if __name__ == '__main__':
    main()

