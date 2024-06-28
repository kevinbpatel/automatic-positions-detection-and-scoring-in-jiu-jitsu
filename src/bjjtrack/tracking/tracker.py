import logging
import pickle
from collections import OrderedDict

import numpy as np

from .visual_matching import JOINTS, Dataset, get_classifier, predict, add_descriptors

CONF_THRESH = 9
def oks(gt, preds, gt_area):
    ious = np.zeros((len(preds), len(gt)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas*2)**2

    xg = gt[:, 0]; yg = gt[:, 1]
    xp = preds[:, 0]; yp = preds[:, 1]
    dx = np.subtract(xg, xp)
    dy = np.subtract(yg, yp)

    e = (dx**2+dy**2)/vars/(gt_area+np.spacing(1))/2
    ious = np.sum(np.exp(-e))/(1.5*e.shape[0]) if len(e) != 0 else 0
    return ious


def get_size(result):
    pose = result['keypoints']
    xmin = pose[:, 0].min()
    xmax = pose[:, 0].max()
    ymin = pose[:, 1].min()
    ymax = pose[:, 1].max()
    w = xmax - xmin
    h = ymax - ymin
    return w * h


def is_background(result, w=640, h=480):
    # filter spurious detections to speed up tracking inference
    return (get_size(result) / (w * h)) < 0.005  or result['keypoints'][:, 2].sum() < CONF_THRESH


def get_width_height_max(pose, pad = 0):
    if pose is None:
        return 0
    xmin = pose[:, 0].min()
    xmax = pose[:, 0].max()
    ymin = pose[:, 1].min()
    ymax = pose[:, 1].max()
    w = xmax - xmin
    h = ymax - ymin
    return np.max((w, h))


def get_base_distance(means):
    distances = []
    for id, mean in means.items():
        distances.append(get_width_height_max(mean))
    return np.power(np.max(distances), 2)


def get_distance(pose1, pose2, index = None):
    if pose1 is None or pose2 is None:
        return -1
    if index is None:
        return np.square(pose1[:, :2] - pose2[:, :2]).mean()
    return np.square(pose1[index, :2] - pose2[index, :2]).mean()


def get_prev_mean(positions, decay = 0.8):
    a = np.zeros(shape=(17, 3))
    weights = []
    #n = len(positions)
    weight = 0.33
    weight_factors = []
    for i, kpts in enumerate(positions):
        weight_factors.append(weight)
        w = kpts[:,2] * weight
        weight = np.power(weight, decay)
        a += (kpts.T * w).T
        weights.append(w)
    weight_mean = np.mean(weight_factors)
    return (a.T / np.sum(weights, axis=0)).T, weight_mean


def intersect_over_area(keypointsA, keypointsB):
    boxA = (keypointsA[:, 0].min(), keypointsA[:, 1].min(),
            keypointsA[:, 0].max(), keypointsA[:, 1].max())
    boxB = (keypointsB[:, 0].min(), keypointsB[:, 1].min(),
            keypointsB[:, 0].max(), keypointsB[:, 1].max())
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    ioaA = interArea / float(boxAArea)
    ioaB = interArea / float(boxBArea)
    return ioaA, ioaB


def set_value(pose_result, val, key):
    max_val = pose_result.get(key, -1)
    if val >= max_val:
        pose_result[key] = val
    else:
        pose_result[key] = max_val


def order_results(pose_results, n=17, unmatched_id=3, img_size=(640, 480)):
    w, h = img_size
    scores = []
    for result in pose_results:
        result['ioa'] = -1
        if 'track_id' not in result:
            result['track_id'] = unmatched_id
        pose = result['keypoints']
        nbest = np.argsort(pose[:, 2])[-n:]
        conf = pose[nbest, 2].sum()
        scores.append(conf)
    scores = np.array(scores)
    scores = scores > CONF_THRESH
    index = np.argsort(scores)[::-1]
    results = []
    for i in index:
        result = pose_results[i]
        if is_background(result, w=w, h=h):
            continue
        results.append(result)
    return results


class Tracker():
    def __init__(self, tracker_path, n, img_size,
                 pose_window_len, dist_tresh,
                 joints, njoints, joint_conf_tresh, 
                 dataset_update_freq, classifier_update_freq, 
                 vis_conf_tresh, unmatched_id = 3):
        
        self.tracker_path = tracker_path
        self.image_size = img_size
        self.pose_anchors = OrderedDict()
        self.pose_anchor_means = dict()
        self.structure_anchors = OrderedDict()
        self.structure_anchor_means = dict()
        self.weight_means = dict()

        self.datasets_updated = dict()

        #ids
        self.ids = list(range(n))
        for id in self.ids:
            self.pose_anchors[id] = []
            self.structure_anchors[id] = []
            self.datasets_updated[id] = -dataset_update_freq

        # pose anchors
        self.njoints = njoints
        self.pose_conf_tresh = 17 * joint_conf_tresh
        self.pose_window_len = pose_window_len
        self.dist_tresh = dist_tresh
        

        # visual datasets
        self.vis_conf_tresh = vis_conf_tresh
        self.dataset_update_freq = dataset_update_freq
        matching_datasets = dict()
        for joint in joints + ['all']:
            new_dataset = Dataset(joint)
            matching_datasets[joint] = new_dataset
        self.matching_datasets = matching_datasets

        self.classifiers = dict()
        self.classifier_update_freq = classifier_update_freq
        self.classifiers_updated = -1

        self.logger = logging.getLogger("tracking")

        # Load tracking model
        with open(self.tracker_path, 'rb') as f:
            self.tracker =  pickle.load(f)

        self.unmatched_id = unmatched_id


    def compute_ioas(self, pose_results):
        for result1 in pose_results:
            id1 = result1.get('vis_id', None)
            if id1 == -1:
                id1 = result1.get('pose_id')
            if id1 not in self.ids:
                continue
            for result2 in pose_results:
                if result2['keypoints'][:, 2].sum() < 8.5:
                    continue
                id2 = result2.get('vis_id', None)
                if id2 == -1:
                    id2 = result2.get('pose_id')
                if id2 not in self.ids:
                    continue
                if id1 != id2:
                    ioa1, ioa2 = intersect_over_area(result1['keypoints'], result2['keypoints'])
                    set_value(result1, ioa1, 'ioa')
                    set_value(result2, ioa2, 'ioa')


    def postprocess(self, matched, pose_results):
        for id1, i1 in matched.items():
            result1 = pose_results[i1]
            for id2, i2 in matched.items():
                if id1 >= id2:
                    continue
                else:
                    result2 = pose_results[i2]
                    ioa1, ioa2 = intersect_over_area(result1['keypoints'], result2['keypoints'])
                    #match_dist = np.square(np.array(result1['structure']) - np.array(result2['structure'])).sum()
                    # match_dist = get_distance(result1['keypoints'], result2['keypoints'])
                    # set_value(result1, match_dist, 'match_dist')
                    # set_value(result2, match_dist, 'match_dist')
                    # self.logger.info(f'match_dist {id1}-{id2}: {match_dist}')
                    
                    result1['ioa'] = ioa1
                    result2['ioa'] = ioa2
                    

    def match_pose(self, pose_result, anchor_means, n=17):
        distances = []
        for id in self.ids:
            prev_mean = anchor_means[id]
            if prev_mean is None:
                distances.append(float('inf'))
                continue
            pose = pose_result['keypoints']
            nbest = np.argsort(pose[:, 2])[-n:]
            dist = get_distance(prev_mean, pose, index=nbest)
            distances.append(dist)

        distances = np.array(distances)
        distances = distances / self.base_dist
        id = int(np.argmin(distances))
        #check if it is to far from closest pose or to close to other poses dont match
        if (distances[id] > self.dist_tresh) or np.sum(distances < self.dist_tresh) > 1:
            id = -1
        self.logger.info(f'distances: {distances} {id}')
        return id, distances


    def match_structure(self, pose_result, anchor_means):
            distances = []
            for id in self.ids:
                prev_mean = anchor_means[id]
                if prev_mean is None:
                    distances.append(float('inf'))
                    continue
                structure = pose_result['structure']
                dist = np.square(structure - prev_mean).sum()
                distances.append(dist)

            distances = np.array(distances)
            distances = distances / self.base_dist
            id = int(np.argmin(distances))
            #check if it is to far from closest pose or to close to other poses dont match
            self.logger.info(f'structures: {distances} {id}')
            return id, distances


    def match_appearance(self, pose_result, n = 3):
        preds = predict(pose_result, self.classifiers, n = 3)
        preds = preds / preds.sum()
        id = int(np.argmax(preds))
        if preds[id] < 0.66:#self.vis_conf_tresh:
            id = -1
        self.logger.info(f'prediction: {preds} {id}')
        return id, preds


    def get_anchor_means(self, frame):
        pose_means = dict()
        struct_means = dict()
        weight_means = dict()
        for id in self.ids:
            pose_anchor = self.pose_anchors[id]
            structure_anchor = self.structure_anchors[id]
            if not len(pose_anchor):
                pose_mean = None
                struct_mean = None
                mean_weights = -1
            else:
                pose_mean, mean_weights = get_prev_mean(pose_anchor)
                struct_mean = np.mean(structure_anchor, 0)
            pose_means[id] = pose_mean
            struct_means[id] = struct_mean
            weight_means[id] = mean_weights
        self.pose_anchor_means[frame] = pose_means
        self.structure_anchor_means[frame] = struct_means
        self.weight_means[frame] = weight_means
        self.base_dist = get_base_distance(pose_means)
        return pose_means, struct_means


    def match(self, pose_results, frame, img = None):
        ordered_results = order_results(pose_results, n=self.njoints, unmatched_id=self.unmatched_id, img_size=self.image_size)
        matched = dict()
        pose_means, structure_means = self.get_anchor_means(frame)

        for i, pose_result in enumerate(ordered_results):
            pose_result['track_id'] = self.unmatched_id
            pose_id, distances = self.match_pose(pose_result, pose_means, n=self.njoints)
            pose_result['pose_id'] = pose_id
            pose_result['distances'] = distances
            struct_id, structures = self.match_structure(pose_result, structure_means)
            pose_result['struct_id'] = struct_id
            pose_result['struct_dist'] = structures
            vis_id, vis_scores = self.match_appearance(pose_result)
            pose_result['vis_id'] = vis_id
            pose_result['vis_score'] = vis_scores

            if pose_id != -1 or vis_id != -1:
                id = -1
                if pose_id == vis_id:
                    id = pose_id
                elif pose_id == -1:
                    if distances[vis_id] != float('inf') and (distances[vis_id] > self.dist_tresh or np.argmin(distances) != vis_id):
                        self.logger.warn('Visual and keypoint matching disagree strongly')
                        continue
                    id = vis_id
                elif vis_id == -1:
                    id = pose_id
                    self.logger.warn('Could not match visually.')
                else:
                    self.logger.warn('Visual and keypoint matching disagree')
                    continue

                conf = pose_result['keypoints'][:, 2].sum()
                if conf < self.pose_conf_tresh:
                    self.logger.info(f'Confidence {conf} lower than threshold {self.pose_conf_tresh}')
                    continue


                if id != -1 and id not in matched:
                    pose_result['track_id'] = id
                    matched[id] = i
        
        self.compute_ioas(ordered_results)
        
        #update anchors and visual datasets
        updated = self.update(matched, ordered_results, frame)
        return matched, ordered_results, updated


    def update_classifiers(self):
        self.logger.info('Updating classifiers')
        for joint, dataset in self.matching_datasets.items():
            self.classifiers[joint] = get_classifier(dataset, method='mlp')


    def update(self, matched, results, frame, init = False, init_freq = 10):
        updated = False
        for id in self.ids:
            if id in matched:
                i = matched[id]
                result = results[i]
                self.add_pose(id, result)
                if self.datasets_updated[id] + self.dataset_update_freq < frame and result['ioa'] < 0.90 or (init and self.datasets_updated[id] + init_freq <= frame):
                    updated = True
                    add_descriptors(result, id, self.matching_datasets, frame)
                    self.datasets_updated[id] = frame
            else:
                self.remove_pose(id)
        if self.classifiers_updated + self.classifier_update_freq < frame or (init and self.classifiers_updated + init_freq <= frame):
            self.update_classifiers()
            self.classifiers_updated = frame
        return updated


    def add_pose(self, id, pose_result):
        pose_anchor = self.pose_anchors[id]
        struct_anchor = self.structure_anchors[id]
        pose_anchor.append(pose_result['keypoints'])
        struct_anchor.append(pose_result['structure'])
        if len(pose_anchor) > self.pose_window_len:
            pose_anchor.pop(0)
            struct_anchor.pop(0)


    def remove_pose(self, id):
        pose_anchor = self.pose_anchors[id]
        struct_anchor = self.structure_anchors[id]
        if len(pose_anchor):
            pose_anchor.pop(0)
            struct_anchor.pop(0)

    
    def save_data(self, path, suffix = ''):
        print('Saving pose means', end = '... ')
        with open(f"{path}/predictions/pose_means{suffix}.pickle", "wb") as f:
            pickle.dump(self.pose_anchor_means, f) 
        print('done.')

        print('Saving structure means', end = '... ')
        with open(f"{path}/predictions/structure_means{suffix}.pickle", "wb") as f:
            pickle.dump(self.structure_anchor_means, f) 
        print('done.')

        print('Saving weight means', end = '... ')
        with open(f"{path}/predictions/weight_means{suffix}.pickle", "wb") as f:
            pickle.dump(self.weight_means, f) 
        print('done.')

        print('Saving visual datasets', end = '... ')
        with open(f"{path}/predictions/features_dataset{suffix}.pickle", "wb") as f:
            pickle.dump(self.matching_datasets, f) 
        print('done.')


    def symulate(self, pose_results, frame):
        ordered_results = order_results(pose_results, n=self.njoints, unmatched_id=self.unmatched_id, img_size=self.image_size)
        matched = dict()
        pose_means, structure_means = self.get_anchor_means(frame)
        p1 = pose_means[0]
        p2 = pose_means[1]
        
        struct_mean1 = structure_means[0]
        struct_mean2 = structure_means[1]
        struct_mean1 = max_pool_1d(struct_mean1, 68)
        struct_mean2 = max_pool_1d(struct_mean2, 68)

        base_dist = get_base_distance(pose_means)
        mean_dist = distance(p1, p2, shape=51)
        if mean_dist[0] != -1:
            mean_dist = mean_dist / base_dist

        for i, pose_result in enumerate(ordered_results):
            distance1 = distance(pose_result['keypoints'], p1, shape=51) 
            if distance1[0] != -1:
                distance1 = distance1 / base_dist
            distance2 = distance(pose_result['keypoints'], p2, shape=51)
            if distance2[0] != -1:
                distance2 = distance2 / base_dist

            #POSE ID
            distances = np.array([distance1.mean(), distance2.mean()])
            distances[distances == -1] = float('inf')
            pose_id = int(np.argmin(distances))
            if (distances[pose_id] > self.dist_tresh) or np.sum(distances < self.dist_tresh) > 1:
                pose_id = -1
            self.logger.info(f'distance: {distances} {pose_id}')
            pose_result['pose_id'] = pose_id
            pose_result['distances'] = distances

            vis1, vis2 = self.predict(pose_result)

            #VIS_ID
            vis = np.array([np.sum(vis1), np.sum(vis2)])
            vis = vis / vis.sum()
            vis_id = int(np.argmax(vis))
            if vis[vis_id] < 0.6:
                vis_id = -1
            self.logger.info(f'prediction: {vis} {vis_id}')
            pose_result['vis_id'] = vis_id
            pose_result['vis_score'] = vis

            struct = max_pool_1d(pose_result['structure'], 68)
            #struct = result['structure']
            #struct1 = distance(result['structure'], sturct_mean1 , shape=len(result['structure']))
            #struct2 = distance(result['structure'], sturct_mean2 , shape=len(result['structure']))
            struct1 = distance(struct, struct_mean1 , shape=68)
            struct2 = distance(struct, struct_mean2 , shape=68)

            sdists = np.array([struct1.mean(), struct2.mean()])
            sdists[sdists == -1] = float('inf')
            
            struct_id = int(np.argmin(sdists))
            self.logger.info(f'structure: {sdists} {struct_id}')
            pose_result['struct_id'] =  struct_id
            pose_result['struct_dist'] = sdists

            id = pose_result['track_id']
            if id in self.ids and id not in matched:
                matched[id] = i
        
        self.compute_ioas(ordered_results)
        self.postprocess(matched, ordered_results)
        
        #update anchors and visual datasets
        self.update(matched, ordered_results, frame)
        return ordered_results


    def predict(self, result):
        votes0 = []
        votes1 = []
        for joint, descriptor in result['descriptors'].items():
            preds = self.classifiers[joint].predict_proba([descriptor])[0]
            votes0.append(preds[0])
            votes1.append(preds[1])
        return votes0, votes1


    def init_tracking(self, pose_results, frame):
        ordered_results = order_results(pose_results, n=self.njoints, unmatched_id=self.unmatched_id, img_size=self.image_size)
        matched = dict()
        pose_means, _ = self.get_anchor_means(frame)
        for i, pose_result in enumerate(ordered_results):
            id, distances = self.match_pose(pose_result, pose_means)
            unassigned = np.where(distances == float('inf'))[0]
            assigned = np.where(distances != float('inf'))[0]
            if len(unassigned):
                id = int(unassigned[0])
                for pid in assigned:
                    if oks(pose_means[pid], pose_result['keypoints'], get_size(pose_result)) > 0.1: ## TODO - HARDCODED
                        id = -1
                if id != -1:
                    pose_means[id] = pose_result['keypoints']
                    self.base_dist = get_base_distance(pose_means)

            if id != -1 and id not in matched:
                pose_result['track_id'] = id
                matched[id] = i
        
        #update anchors and visual datasets
        updated = self.update(matched, ordered_results, frame, init=True)
        return matched, ordered_results, updated
     

    def track(self, pose_results, frame):
        ordered_results = order_results(pose_results, n=self.njoints, unmatched_id=self.unmatched_id, img_size=self.image_size)

        pose_means, structure_means = self.get_anchor_means(frame)
        weight0 = self.weight_means[frame][0]
        weight1 = self.weight_means[frame][1]
        p1 = pose_means[0]
        p2 = pose_means[1]
        
        struct_mean1 = structure_means[0]
        struct_mean2 = structure_means[1]
        struct_mean1 = max_pool_1d(struct_mean1, 68)
        struct_mean2 = max_pool_1d(struct_mean2, 68)

        
        base_dist = get_base_distance(pose_means)
        mean_dist = distance(p1, p2, shape=51)
        if mean_dist[0] != -1:
            mean_dist = mean_dist / base_dist

        features0 = []
        features1 = []
        if not len(ordered_results):
            return dict(), pose_results, False
        for pose_result in ordered_results:
            pose_result['track_id'] = self.unmatched_id
            distance1 = distance(pose_result['keypoints'], p1, shape=51) 
            if distance1[0] != -1:
                distance1 = distance1 / base_dist
            distance2 = distance(pose_result['keypoints'], p2, shape=51)
            if distance2[0] != -1:
                distance2 = distance2 / base_dist

            #POSE ID
            distances = np.array([distance1.mean(), distance2.mean()])
            distances[distances == -1] = float('inf')
            pose_id = int(np.argmin(distances))
            if (distances[pose_id] > self.dist_tresh) or np.sum(distances < self.dist_tresh) > 1:
                pose_id = -1
            self.logger.info(f'distance: {distances} {pose_id}')
            pose_result['pose_id'] = pose_id
            pose_result['distances'] = distances

            vis1, vis2 = self.predict(pose_result)

            #VIS_ID
            vis = np.array([np.sum(vis1), np.sum(vis2)])
            vis = vis / vis.sum()
            vis_id = int(np.argmax(vis))
            if vis[vis_id] < 0.6:
                vis_id = -1
            self.logger.info(f'prediction: {vis} {vis_id}')
            pose_result['vis_id'] = vis_id
            pose_result['vis_score'] = vis

            struct = max_pool_1d(pose_result['structure'], 68)
            struct1 = distance(struct, struct_mean1 , shape=68)
            struct2 = distance(struct, struct_mean2 , shape=68)

            sdists = np.array([struct1.mean(), struct2.mean()])
            sdists[sdists == -1] = float('inf')
            
            struct_id = int(np.argmin(sdists))
            self.logger.info(f'structure: {sdists} {struct_id}')
            pose_result['struct_id'] =  struct_id
            pose_result['struct_dist'] = sdists

            conf = pose_result['keypoints'][:, 2]

            feature0 = np.concatenate([distance1, vis1, struct1, conf, mean_dist, [weight0]])
            feature1 = np.concatenate([distance2, vis2, struct2, conf, mean_dist, [weight1]])
            features0.append(feature0)
            features1.append(feature1)
        
        
        self.compute_ioas(ordered_results)

        tracking_predictions = dict()
        tracking_predictions[0] = self.tracker.predict_proba(features0)
        tracking_predictions[1] = self.tracker.predict_proba(features1)

        tracking_predictions = np.stack((tracking_predictions[0][:, 1],tracking_predictions[1][:, 1]))

        matched = dict()
        for id in self.ids:
            predictions = tracking_predictions[id, :]
            opposing = tracking_predictions[~id, :]
            self.logger.info(f'tracking - {id} : {predictions}')
            best_ids = np.argsort(predictions)[::-1]
            for best in best_ids:
                matched_result = ordered_results[best]
                #print(f'ID-{id}: {predictions[best]}, res:{best},conf:{matched_result["keypoints"][:, 2].sum()}')
                if matched_result['keypoints'][:, 2].sum() < CONF_THRESH:
                    continue
                if predictions[best] > 0.85 and opposing[best] < 0.2:
                    matched_result['track_id'] = id
                    matched_result['track_prob'] = predictions[best]
                    matched[id] = best
                    break

        self.postprocess(matched, pose_results) # only used to create tracking dataset

        updated = self.update(matched, ordered_results, frame)
        return matched, ordered_results, updated


def distance(vec1, vec2, shape = 51):
    if vec1 is None or vec2 is None:
        return np.ones(shape) * -1
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.square(v1 - v2).flatten()


def max_pool_1d(a, to, axis=1):
    if a is None:
        return a
    n = len(a)
    shape = (to, int(n / to))
    return np.array(a).reshape(shape).max(axis)