import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms


pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Resize((256, 192))
])

METHODS = ['knn', 'svc', 'sgd', 'ada', 'mlp', 'rfc']
KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
             'right_knee', 'left_ankle', 'right_ankle']

JOINTS = [KEYPOINTS.index('left_shoulder'), KEYPOINTS.index('right_shoulder'),
          KEYPOINTS.index('left_hip'),      KEYPOINTS.index('right_hip'),
          KEYPOINTS.index('left_knee'),     KEYPOINTS.index('right_knee')]
ALL_JOINTS = [KEYPOINTS.index(keypoint) for keypoint in KEYPOINTS]


def get_features(model, x):
    B, C, H, W = x.shape
    x, (Hp, Wp) = model.patch_embed(x)

    xs = x + model.pos_embed[:, 1:] + model.pos_embed[:, :1]
    
    xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
    xs = xs.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
    xp = xp.detach().cpu().numpy() 
    xs = xs.detach().cpu().numpy()
    return xp, xs


def crop_and_resize(image, keypoints, pad = 0.2):
    H, W, C = image.shape
    xmin = keypoints[:, 0].min()
    xmax = keypoints[:, 0].max()
    ymin = keypoints[:, 1].min()
    ymax = keypoints[:, 1].max()
    w = xmax - xmin
    h = ymax - ymin
    ypad = int(w * pad)
    xpad = int(h * pad)
    xmin = np.max((0, xmin - xpad)).astype(int)
    xmax = np.min((W, xmax + xpad)).astype(int)
    ymin = np.max((0, ymin - ypad)).astype(int)
    ymax = np.min((H, ymax + ypad)).astype(int)
    return cv2.resize(image[ymin:ymax, xmin:xmax, :], (192, 256))


def add_featuremaps(model, image, pose_results, device, joints = None):
    imgs = []
    for pose_result in pose_results:
        img_resized = crop_and_resize(image, pose_result['keypoints'])
        imgs.append(pose_transform(img_resized))
    inputs = torch.stack(imgs).to(device, dtype=torch.float)
    with torch.no_grad():
        featuremaps, structuremaps = get_features(model.backbone, inputs)
    for pose_result, featuremap, structuremap in zip(pose_results, featuremaps, structuremaps):
        kpts = resize_keypoints(pose_result, featuremap)
        pose_result['descriptors'] = get_descriptors(featuremap, kpts, joints = joints)
        structure_descriptors = get_descriptors(structuremap, kpts, joints = ALL_JOINTS)
        pose_result['structure'] = structure_descriptors['all']
    return pose_results


def get_descriptors(featuremap, keypoints, joints = None):
    descriptors = dict()
    all_descriptors = []
    if joints is None:
        joints = JOINTS
    for joint in joints:
        descriptor = get_descriptor(keypoints[joint], featuremap).tolist()
        all_descriptors += descriptor
        descriptors[joint] = descriptor
    descriptors['all'] = all_descriptors
    return descriptors


def get_descriptor(keypoint, featuremap):
    C, H, W = featuremap.shape
    x = np.max((np.min((keypoint[0], W-1)), 0)).astype(int)
    y = np.max((np.min((keypoint[1], H-1)), 0)).astype(int)
    return featuremap[:, y, x]


def init_classifier(method):
    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5, weights = 'distance', metric='euclidean')
    elif method == 'svc':
        clf = SVC(gamma='auto', probability=True, class_weight = 'balanced')
    elif method == 'mlp':
        clf = MLPClassifier(max_iter = 2000, hidden_layer_sizes=(136, 17))
    elif method == 'ada':
        clf = AdaBoostClassifier(n_estimators=200)  
    elif method == 'rfc':
        clf = RandomForestClassifier(n_estimators=200)
    elif method == 'sgd':
        clf = SGDClassifier(max_iter=200, loss='log_loss')
    elif method == 'best':
        clf = dict()
        for m in METHODS:
            clf[m] = init_classifier(m)
    else:
        assert False, f'{method} not supported'
    return clf


def fit(clf, X, y):
    if type(clf) == dict:
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        scores = dict()
        for method, classifier in clf.items():
            print(f'Fitting {method}...', end= ' ')
            classifier.fit(X_train, y_train)
            print('Done')
            scores[method] = classifier.score(X_val, y_val)
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for score in scores:
            print(score)
        best = scores[0][0]
        print(best)
        classifier = clf[best]
        classifier.fit(X, y)
        return classifier
    X = np.array(X)
    y = np.array(y)
    X1 = X[y == 0]
    y1 = y[y == 0]
    X2 = X[y == 1]
    y2 = y[y == 1]
    X3 = X[y == 2]
    y3 = y[y == 2]
    if len(X1) > len(X2):
        idx = np.random.randint(0, high=len(X1), size=(len(X2)), dtype=int)
        X1 = X1[idx]
        y1 = y1[idx]
    elif len(X2) > len(X1):
        idx = np.random.randint(0, high=len(X2), size=len(X1), dtype=int)
        X2 = X2[idx]
        y2 = y2[idx]
    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((y1, y2, y3))
    
    clf.fit(X, y)
    return clf


def get_classifier(dataset, method = 'svc'):
    X = dataset.X
    y = dataset.y
    if not len(X):
        assert False, 'Empty dataset'
    clf = init_classifier(method)
    clf = fit(clf, X, y)
    return clf


def resize_keypoints(pose_result, featuremap, pad = 0.2):
    C, H, W = featuremap.shape
    keypoints = pose_result['keypoints']
    xmin = keypoints[:, 0].min()
    xmax = keypoints[:, 0].max()
    ymin = keypoints[:, 1].min()
    ymax = keypoints[:, 1].max()
    w = xmax - xmin
    h = ymax - ymin
    ypad = int(w * pad)
    xpad = int(h * pad)
    xmin = np.max((0, xmin - xpad))
    xmax = xmax + xpad
    ymin = np.max((0, ymin - ypad))
    ymax = ymax + ypad
    w = xmax - xmin
    h = ymax - ymin
    kpts = keypoints.copy()
    kpts[:, :2] = kpts[:, :2] - np.array([xmin, ymin])
    kpts[:, 0] = kpts[:, 0] / w * W
    kpts[:, 1] = kpts[:, 1] / h * H
    return kpts


def predict(pose_result, classifiers, n = 17):
    descriptors = pose_result['descriptors']
    votes = []
    for joint, descriptor in descriptors.items():
        pred = classifiers[joint].predict_proba([descriptor])[0]
        votes.append(pred)
    votes = np.array(votes)
    maxvals = votes.max(1)
    best = np.argsort(maxvals)[-n:]
    return votes[best].sum(0)


def add_descriptors(pose_result, label, datasets, frame = None):
    descriptors = pose_result['descriptors']
    for joint, descriptor in descriptors.items():
        datasets[joint].X.append(descriptor)
        datasets[joint].y.append(label)
        datasets[joint].ioa.append(pose_result['ioa'])
        if frame:
            datasets[joint].frames.append(frame)


class Dataset():
    def __init__(self, joint):
        self.joint = joint
        self.X = list()
        self.y = list()
        self.ioa = list()
        self.frames = list()

