import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


from bjjtrack.jiujitsu.utils import normalize, get_frame_delay

POSITIONS = ['5050_guard', 'back1', 'back2', 'closed_guard1', 'closed_guard2',
             'half_guard1', 'half_guard2', 'mount1', 'mount2', 'open_guard1', 
             'open_guard2', 'side_control1', 'side_control2', 'standing', 
             'takedown1', 'takedown2', 'turtle1', 'turtle2']


def get_positions(position, cam, suffix = ''):
    predictions_path = f'outputs/{position}-{cam}/predictions/predictions{suffix}.pickle'
    positions_path = f'data/positions/{position}_{cam}_positions.pickle'
    if not os.path.isfile(predictions_path) or not os.path.isfile(positions_path):
        return []
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    with open(positions_path, 'rb') as f:
        positions = pickle.load(f)
    data = []
    for frame, matches in predictions.items():
        if frame not in positions or frame not in predictions:
            continue
        else:
            pos = positions[frame]
        if pos == 'transition':
            continue
        p1 = matches.get(1, None)
        p2 = matches.get(2, None)
        data.append((p1, p2, pos, frame))
    return data


def get_mean_acc(probs, y, frame, window = 6):
    n = window/2
    preds = []
    gt = []
    mean_probs = []
    for frame in frames:
        i = np.where(frames == frame)
        prob_s = probs[(frames - n < frame) & (frames + n > frame)]
        prob_s = prob_s.sum(0)
        mean_probs.append(prob_s)
        pred = POSITIONS[np.argmax(prob_s)]
        gt.append(y[i])
        preds.append(pred)
    mean_probs = np.array(mean_probs)
    return accuracy_score(gt, preds), mean_probs


def multi_angle_predictions(predictions, angles = ['00', '45', '90'], key = 'probs'):
    ref = predictions[angles[0]]
    preds = []
    gt = ref['gt']
    for i, frame in enumerate(ref['frames']):
        probs = [ref[key][i]]
        for angle in angles[1:]:
            frame2 = frame + get_frame_delay(fps[angle], fps[angles[0]], frame)
            j = np.where(predictions[angle]['frames'] == frame2)[0]
            if len(j):
                probs.append(predictions[angle][key][j].flatten())
        s = np.sum(probs, 0)
        pred = POSITIONS[np.argmax(s)]
        preds.append(pred)
    preds = np.array(preds)
    preds_agg = [aggregate(pos) for pos in preds]
    gt_agg = [aggregate(pos) for pos in gt]
    return gt, preds, preds_agg, gt_agg

def get_multi_angle(predictions, angles = ['00', '45', '90'], key = 'probs'):
    gt, preds, preds_agg, gt_agg = multi_angle_predictions(predictions, angles, key)
    return accuracy_score(gt, preds), accuracy_score(preds_agg, gt_agg)


def aggregate(pos):
    if 'guard' in pos:
        if '1' in pos:
            return 'guard1'
        elif '2' in pos:
            return 'guard2'
        else:
            return pos
    return pos


# TODO add args and make script
print('Loading classifier', end ='... ', flush = True)

with open('checkpoints/jiujitsu/classifier.pickle', 'rb') as f:
    clf = pickle.load(f)
print('Done')

angles = ['00',  '45', '90']
fps = {'00': 29.799, 
       '45': 29.992, 
       '90': 30.005}
predictions = dict()

POSITION = 'sparring'
print(POSITION)
results = dict()
for angle in angles:
    positions = get_positions(POSITION, angle, suffix = '_matched')
    X = []
    y = []
    frames = []
    for kpts1, kpts2, position, frame in positions:
        if kpts1 is None or kpts2 is None:
            continue
        p1, p2 = normalize((kpts1['keypoints'], kpts2['keypoints']))
        X.append(p1.flatten().tolist() + p2.flatten().tolist())
        y.append(position)
        frames.append(frame)
    X = np.array(X)
    y = np.array(y)
    frames = np.array(frames)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    accuracy = accuracy_score(preds, y)
    preds_agg = [aggregate(pos) for pos in preds]
    gt_agg = [aggregate(pos) for pos in y]
    accuracy_agg = accuracy_score(preds_agg, gt_agg)
    f1 = f1_score(preds, y, average='weighted')
    mean_acc, mean_probs = get_mean_acc(probs, y, frames, window=15)
    # for i in range(2, 30):
    #     acc = get_mean_acc(probs, y, frames, window=i)
    #     if acc > mean_acc:
    #         mean_acc = acc
    #     print(f"window_len {i}: {acc}")
    predictions[angle] = {'probs' :probs, 
                          'frames': frames,
                          'gt': y,
                          'mean_probs': mean_probs}
    print(f'{angle}, acc: {accuracy} f1: {f1} acc-mean: {mean_acc} ({accuracy_agg}) {len(positions)}')
    results[angle] = {'accuracy': accuracy, 'mean_acc': mean_acc, 'accuracy_agg': accuracy_agg}

two_angle = []
two_angle_agg = []
two_angle_mean = []
two_angle_mean_agg = []
a = ['00', '45']
multi_acc, multi_agg = get_multi_angle(predictions, angles = a)
multi_acc_mean, multi_agg_mean = get_multi_angle(predictions, angles = a, key = 'mean_probs')
print(f'{a} acc: {multi_acc} ({multi_agg}) acc-mean: {multi_acc_mean} ({multi_agg_mean})')
two_angle.append(multi_acc)
two_angle_agg.append(multi_agg)
two_angle_mean.append(multi_acc_mean)
two_angle_mean_agg.append(multi_acc_mean)


a = ['00', '90']
multi_acc, multi_agg = get_multi_angle(predictions, angles = a)
multi_acc_mean, multi_agg_mean = get_multi_angle(predictions, angles = a, key = 'mean_probs')
print(f'{a} acc: {multi_acc} ({multi_agg}) acc-mean: {multi_acc_mean} ({multi_agg_mean})')
two_angle.append(multi_acc)
two_angle_agg.append(multi_agg)
two_angle_mean.append(multi_acc_mean)
two_angle_mean_agg.append(multi_agg_mean)

a = ['45', '90']
multi_acc, multi_agg = get_multi_angle(predictions, angles = a)
multi_acc_mean, multi_agg_mean = get_multi_angle(predictions, angles = a, key = 'mean_probs')
print(f'{a} acc: {multi_acc} ({multi_agg}) acc-mean: {multi_acc_mean} ({multi_agg_mean})')
two_angle.append(multi_acc)
two_angle_agg.append(multi_agg)
two_angle_mean.append(multi_acc_mean)
two_angle_mean_agg.append(multi_agg_mean)


three_angle = []
three_angle_agg = []
three_angle_mean = []
three_angle_mean_agg = []
a = ['00', '45', '90']
multi_acc, multi_agg = get_multi_angle(predictions, angles = a)
multi_acc_mean, multi_agg_mean = get_multi_angle(predictions, angles = a, key = 'mean_probs')
print(f'{a} acc: {multi_acc} ({multi_agg}) acc-mean: {multi_acc_mean} ({multi_agg_mean})')
three_angle.append(multi_acc)
three_angle_agg.append(multi_agg)
three_angle_mean.append(multi_acc_mean)
three_angle_mean_agg.append(multi_agg_mean)


print(f'2: acc:{np.mean(two_angle)} ({np.mean(two_angle_agg)}) acc-mean:{np.mean(two_angle_mean)} ({np.mean(two_angle_mean_agg)})')
print(f'3: acc:{np.mean(three_angle)} ({np.mean(three_angle_agg)}) acc-mean:{np.mean(three_angle_mean)} ({np.mean(three_angle_mean_agg)})')

# Method | One View | Two Views | Three Views
df = pd.DataFrame(columns=['Method', 'One View', 'Two Views', 'Three Views'])
# One View
two_person = []
two_person_avg = []
two_person_agg = []
for angle in ['00', '45', '90']:
    two_person.append(results[angle]['accuracy'])
    two_person_avg.append(results[angle]['mean_acc'])
    two_person_agg.append(results[angle]['accuracy_agg'])
df['One View'] = [f'{np.mean(accuracy).round(2)} ± {np.std(accuracy).round(3)}' for accuracy in [two_person, two_person_avg, two_person_agg]]

# Two Views
df['Two Views'] = [f'{np.mean(accuracy).round(2)} ± {np.std(accuracy).round(3)}' for accuracy in [two_angle, two_angle_mean, two_angle_agg]]

# Three Views
# Bootstrap the results to obtain uncertainty
m = 10
gt, preds, _, _ = multi_angle_predictions(predictions, angles = ['00', '45', '90'], key = 'probs')
gt_mean, preds_mean, preds_agg_mean, gt_agg_mean = multi_angle_predictions(predictions, angles = ['00', '45', '90'], key = 'mean_probs')
three_angle = []
three_angle_avg = []
three_angle_mean_agg = []
for i in range(m):
    idx = np.random.choice(len(gt), len(gt))
    acc = accuracy_score(np.array(preds)[idx], np.array(gt)[idx])
    acc_avg = accuracy_score(np.array(preds_mean)[idx], np.array(gt)[idx])
    acc_agg_avg = accuracy_score(np.array(preds_agg_mean)[idx], np.array(gt_agg_mean)[idx])
    three_angle.append(acc)
    three_angle_avg.append(acc_avg)
    three_angle_mean_agg.append(acc_agg_avg)

df['Three Views'] = [f'{np.mean(accuracy).round(2)} ± {np.std(accuracy).round(3)}' for accuracy in [three_angle, three_angle_avg, three_angle_mean_agg]]

# MEthod
df['Method'] = ['Two Person', 'Two Person*', 'Two person ★']

print(df.to_markdown(index=False, tablefmt='github'))