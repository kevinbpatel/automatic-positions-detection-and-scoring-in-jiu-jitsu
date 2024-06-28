import os
import pickle
import argparse

import cv2
import numpy as np

from bjjtrack.jiujitsu.scoring import BJJudge
from bjjtrack.jiujitsu.utils import normalize, get_frame_delay
from bjjtrack.jiujitsu.vis import POSITIONS, visualize, visualize_predictions, visualize_results

def frames(start, end):
    return list(range(start, end))

ANGLES = {
    '00' : frames(340, 568) + frames(608, 704) + frames(1190, 1565) + frames(2210, 2345) +
           frames(2376, 2492) + frames(2613, 2908) + frames(3316, 3506) + frames(5063, 5173) +
           frames(5449, 5697) + frames(6061, 6185) + frames(6432, 6515) + frames(8380, 8419),
    '45' : frames(0, 340) + frames(568, 608) + frames(838, 896) + frames(1084, 1190) +
           frames(1966, 2210) + frames(2492, 2613) + frames(3162, 3316) + frames(3506, 3689) +
           frames(3830, 4153) + frames(5173, 5306) + frames(6185, 6432) + frames(7373, 7960) +
           frames(8100, 8380) + frames(8419, 10000),
}
COLORS = [(0, 0, 255), (255, 0, 0)]
DECAY = 0.95

parser = argparse.ArgumentParser(description='Compile scoring videos.')
parser.add_argument('position', type = str)
args = parser.parse_args()

sequence = args.position
def get_position_predictions(preds, img, prev, frame, r=2, t=1, reverse = True):
    p1 = None
    p2 = None
    if frame in preds:
        matches = preds[frame]
        p1 = matches.get(1, None)
        p2 = matches.get(2, None)
    if p1:
        visualize(img, p1['keypoints'].T, COLORS[0], radius=r, thickness=t)
    if p2:
        visualize(img, p2['keypoints'].T, COLORS[1], radius=r, thickness=t)
    if p1 and p2:
        kpts1, kpts2 = normalize((p1['keypoints'], p2['keypoints']))
        x = kpts1.flatten().tolist() + kpts2.flatten().tolist()
        return clf.predict_proba([x])[0]
    return prev * DECAY


with open(f'data/positions/{sequence}_00_positions.pickle', 'rb') as f:
    positions = pickle.load(f)
# TODO add to args
with open('checkpoints/jiujitsu/classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

with open(f'outputs/{sequence}-00/predictions/predictions_matched.pickle', 'rb') as f:
    preds1 = pickle.load(f)

with open(f'outputs/{sequence}-45/predictions/predictions_matched.pickle', 'rb') as f:
    preds2 = pickle.load(f)

with open(f'outputs/{sequence}-90/predictions/predictions_matched.pickle', 'rb') as f:
    preds3 = pickle.load(f)

if sequence == 'sparring':
    cap1 = cv2.VideoCapture(f'data/videos/{sequence}2_00.mp4')
    cap2 = cv2.VideoCapture(f'data/videos/{sequence}2_45.mp4')
    cap3 = cv2.VideoCapture(f'data/videos/{sequence}2_90.mp4')
else:
    cap1 = cv2.VideoCapture(f'data/videos/{sequence}_00.mp4')
    cap2 = cv2.VideoCapture(f'data/videos/{sequence}_45.mp4')
    cap3 = cv2.VideoCapture(f'data/videos/{sequence}_90.mp4')

fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
fps3 = cap3.get(cv2.CAP_PROP_FPS)

size = (640, 352)
# TODO: separate scoring evaluation and creation of video to make it run faster
videoWriter = cv2.VideoWriter(f'outputs/scoring/{sequence}_scoring.mp4', 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                30, size)

judge = BJJudge(player1 = 'red', player2 = 'blue', verbose = True)
judge_gt = BJJudge(player1 = 'red', player2 = 'blue')
pos_pred1 = np.zeros(18)
pos_pred2 = np.zeros(18)
pos_pred3 = np.zeros(18)
position = 'standing'


pred_positions = []
gt_positions = []
for frame in range(200, 9740): # 
    if frame in positions:
        gt_pos = positions[frame]
    else:
        gt_pos = 'transition'
    gt_positions.append([frame, gt_pos])

    frame2 = frame + get_frame_delay(fps2, fps1, frame)
    frame3 = frame + get_frame_delay(fps3, fps1, frame)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, frame3)
    
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    ret3, img3 = cap3.read()

    if not ret1 or not ret2 or not ret3:
        break

    pos_pred1 = get_position_predictions(preds1, img1, pos_pred1, frame, r=4, t=2)
    pos_pred2 = get_position_predictions(preds2, img2, pos_pred2, frame2, r=4, t=2)
    pos_pred3 = get_position_predictions(preds3, img3, pos_pred3, frame3, r=2, t=1)

    img1 = cv2.resize(img1, (640, 352))
    img2 = cv2.resize(img2, (640, 352))
    img3 = cv2.resize(img3, (640, 352))

    visualize_predictions(pos_pred1, img1, gt_pos)
    visualize_predictions(pos_pred2, img2, gt_pos)
    visualize_predictions(pos_pred3, img3, gt_pos)

    if frame in ANGLES['00'] and False:
        img = img1
    elif frame in ANGLES['45'] or True:
        img = img2
    else:
        img = img3
    # img = np.vstack((img1, img2, img3))

    pos_all = np.sum([pos_pred1, pos_pred2, pos_pred3], 0)
    sum = pos_all.sum()
    if sum:
        pos_all = pos_all / sum
    conf = pos_all.max()
    if conf > 0.5:
        pred_pos = POSITIONS[np.argmax(pos_all)]
    else:
        pred_pos = 'transition'
    pred_positions.append([frame, pred_pos, conf])
    judge.update(pred_pos, frame)
    judge_gt.update(gt_pos, frame)
   
    visualize_results(img, judge, match = 'PREDICTED', x = 620, y = 30, color1 = COLORS[0], color2 = COLORS[1])
    #visualize_results(img, judge_gt, match = 'ACTUAL', x = 620, y = 85 , color1 = COLORS[0], color2 = COLORS[1])
    visualize_predictions(pos_all, img, gt_pos, y=250)#, x = 400)
    
    videoWriter.write(img)

s1, s2 = judge.get_result()
s1_gt, s2_gt = judge_gt.get_result()
print('red:', s1, s1_gt)
print('blue:', s2, s2_gt)

gt_trace = judge_gt.get_trace()
pred_trace = judge.get_trace()

stats = {
    'gt_scoring': gt_trace,
    'gt_pos': gt_positions,
    'pred_scoring': pred_trace,
    'pred_pos': pred_positions
}

os.makedirs('outputs/scoring', exist_ok=True)
with open(f'outputs/scoring/{sequence}_scoring.pickle', 'wb') as f:
    pickle.dump(stats, f)