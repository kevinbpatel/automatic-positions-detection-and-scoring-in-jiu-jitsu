import numpy as np

COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

COCO_LR_SWITCH = [
        'nose',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear',
        'right_shoulder',
        'left_shoulder',
        'right_elbow',
        'left_elbow',
        'right_wrist',
        'left_wrist',
        'right_hip',
        'left_hip',
        'right_knee',
        'left_knee',
        'right_ankle',
        'left_ankle'
    ]

COCO_LR_INDEX = [COCO_LR_SWITCH.index(joint) for joint in COCO_KP_ORDER]

def normalize(keypoints):
    coords = np.concatenate(keypoints) #(kpts1, kpts2)
    xmin = coords[:, 0].min()
    ymin = coords[:, 1].min()
    coords[:, 0] = coords[:, 0] - xmin
    coords[:, 1] = coords[:, 1] - ymin
    xmax = coords[:, 0].max()
    ymax = coords[:, 1].max()
    max = np.max([xmax, ymax])
    coords[:, :2] = coords[:, :2] / max
    kpts_n1 = coords[:17]
    kpts_n2 = coords[17:]
    return kpts_n1, kpts_n2


def flip(pose1, pose2):
    coords = np.concatenate((pose1, pose2))
    xmin = coords[:,0].min()
    xmax = coords[:,1].max()
    w = xmax - xmin
    coords[:, 0] = w - coords[:, 0]
    #set min to 0
    xmin = coords[:,0].min()
    coords[:, 0] = coords[:, 0] - xmin
    flipped1 = coords[:17]
    flipped2 = coords[17:]
    return flipped1[COCO_LR_INDEX], flipped2[COCO_LR_INDEX]


def rotate(p, origin=(0, 0), theta = 15):
    A = np.matrix([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    w = np.zeros(p.shape)
    p_shifted = p - np.array(origin)
    for i,v in enumerate(p_shifted):
        w[i] = A @ v
    return w


def random_rotate(pose1, pose2, degrees = 15):
    coords = np.concatenate((pose1, pose2))
    xmin = coords[:,0].min()
    xmax = coords[:,1].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    x = (xmax-xmin) / 2
    y = (ymax-ymin) / 2
    degrees = np.random.random() * degrees
    neg = np.random.random() > 0.5
    if neg:
        degrees = -degrees
    angle = np.deg2rad(degrees)
    coords[:,:2] = rotate(coords[:,:2], (x,y), angle)
    xmin = coords[:,0].min()
    ymin = coords[:, 1].min()
    coords[:, 0] = coords[:, 0] - xmin
    coords[:, 1] = coords[:, 1] - ymin
    rot1 = coords[:17]
    rot2 = coords[17:]
    return rot1, rot2

def add_random_noise(kpts1, kpts2, s=0.05):
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) * s
    x_noise1 = np.random.normal(0, sigmas, kpts1.shape[0])
    y_noise1 = np.random.normal(0, sigmas, kpts1.shape[0])
    x_noise2 = np.random.normal(0, sigmas, kpts2.shape[0])
    y_noise2 = np.random.normal(0, sigmas, kpts2.shape[0])
    kpts1[:, 0] += x_noise1
    kpts1[:, 1] += y_noise1
    kpts2[:, 0] += x_noise2
    kpts2[:, 1] += y_noise2
    return kpts1, kpts2

    
def get_position(frame, position_times):
    if frame in position_times:
        return position_times[frame]
    return "transition"


def get_frame_delay(fps, fps_ref, frame_count):
    fps_factor = fps/fps_ref
    real_frame = fps_factor * frame_count
    delay = int(np.round(real_frame - frame_count))
    return delay


def single_to_double(predictions, id):
    if id == 1:
        return predictions[[0, 2, 1, 3, 6, 3, 6, 5, 4, 3, 6, 8, 7, 9, 11, 10, 13, 12]]
    elif id == 2: 
        return predictions[[0, 1, 2, 6, 3, 6, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12, 13]]
    assert False, f"Wrong id {id}"
