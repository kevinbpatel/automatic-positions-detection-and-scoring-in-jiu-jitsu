import cv2

POSITIONS = ['5050_guard', 'back1', 'back2', 'closed_guard1', 'closed_guard2',
             'half_guard1', 'half_guard2', 'mount1', 'mount2', 'open_guard1', 'open_guard2',
             'side_control1', 'side_control2', 'standing', 'takedown1', 'takedown2',
             'turtle1', 'turtle2']

COCO_KP_ORDER = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
             'right_knee', 'left_ankle', 'right_ankle', 'center']

def prettify(position, one='- blue', two = ' - red'):
    pretty = position.replace('_', ' ').replace('2', two).replace('1', one)
    return pretty.capitalize()
    
    
def visualize_predictions(pred, img, gt, bar_color = (0, 0, 255), x = 20, y = 200, pretty=True):
    first = True
    for pred, pos in sorted(zip(pred, POSITIONS), reverse=True)[:5]:
        if first:
            if gt == 'transition':
                color = (255, 0, 0)
            elif pos == gt:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            first = False
        else:
            color = (255, 255, 255)
        if pretty:
            cv2.putText(img, prettify(pos), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                color=color,thickness=1)
        else:
            cv2.putText(img, pos, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                    color=color,thickness=1)
        cv2.rectangle(img, (x + 120, y), (x + 220, y - 10), (255, 255, 255), -1)
        cv2.rectangle(img, (x + 120, y), (x + 120 + int(100*pred), y - 10), bar_color, -1)
        cv2.putText(img, f"{float(pred):.03}", (x + 120, y - 1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                color=(0, 0, 0),thickness=1)
        y += 20
    
    cv2.putText(img, gt, (x, y - 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
               color=(0, 0, 0),thickness=1)


def visualize_results(img, judge, match = 'qualifiers', x = 120, y = 120, height = 20, color1 = (255, 0, 0), color2 = (0, 0, 255)):
    #match name eg. quarter finals
    cv2.putText(img, match, (x-100, y-23), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, 
                color=(0, 0, 0), thickness=1)


    points1 = str(judge.points1)
    player1 = judge.player1
    cv2.rectangle(img, (x-100, y - height), (x, y), (255, 255, 255), -1)
    cv2.putText(img, player1, (x-95, int(y - height*0.25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                color=color1, thickness=1)
    cv2.putText(img, points1, (x-25, int(y - height*0.25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, (x-100, y), (x,  y - height), (0, 0, 0))
    cv2.rectangle(img, (x-100, y), (x-30, y-height), (0, 0, 0))
    
    points2 = str(judge.points2)
    player2 = judge.player2
    cv2.rectangle(img, (x-100, y), (x, y + height), (255, 255, 255), -1)
    cv2.putText(img, player2, (x-95, int(y + height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                color= color2, thickness=1)
    cv2.putText(img, points2, (x-25, int(y + height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, (x-100, y), (x, y + height), (0, 0, 0))
    cv2.rectangle(img, (x-100, y), (x-30, y + height), (0, 0, 0))


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_shoulder')],
        [keypoints.index('left_hip'), keypoints.index('left_shoulder')]
    ]
    return kp_lines

COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)


def visualize(img, keypoints, color, radius = 4, thickness = 2):
    dataset_keypoints = COCO_KP_ORDER
    kp_lines = COCO_KP_CONNECTIONS 

    line_thickness = thickness  

    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = int(keypoints[0, i1]), int(keypoints[1, i1])
        p2 = int(keypoints[0, i2]), int(keypoints[1, i2])
        img = cv2.line(
            img, p1, p2,
            color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
        img = cv2.circle(
            img, p1,
            radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
        img = cv2.circle(
            img, p2,
            radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
        c1 = keypoints[2, i1]
        c2 = keypoints[2, i2]
        # if c1 < 0.3:
        #     cv2.putText(img, f"{keypoints[2, i1]:.04}", (p1), 
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # if c2 < 0.3:
        #     cv2.putText(img, f"{keypoints[2, i2]:.04}", (p2), 
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img