import pickle


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches

sns.set_theme()

with open('outputs/scoring/sparring_scoring.pickle', 'rb') as f:
    scoring = pickle.load(f)



gt_scoring = scoring.get('gt_scoring')
pred_scoring = scoring.get('pred_scoring')
gt_pos = np.array(scoring.get('gt_pos'))
pred_pos = np.array(scoring.get('pred_pos'))

gts = np.array(gt_scoring)
gts1 = gts[gts[:, 2] == 0]
gts2 = gts[gts[:, 2] == 1]

pts = np.array(pred_scoring)
pts1 = pts[pts[:, 2] == 0]
pts2 = pts[pts[:, 2] == 1]

def get_lines(pos, position):
    frames = pos[pos[:, 1] == position, 0].astype(int)
    lines = []
    if len(frames):
        line = [frames[0], frames[0]]
        for frame in frames[1:]:
            if frame - 1 == line[1]:
                line[1] = frame
            else:
                lines.append(line)
                line = [frame, frame]
        lines.append(line)
        return lines
    return []

positions = [
    ('5050_guard', -1), 
    ('standing', 1),
    ('takedown1', 2),
    ('takedown2', -2),
    ('open_guard1', 3), 
    ('open_guard2', -3),
    ('closed_guard1', 4),
    ('closed_guard2', -4),
    ('half_guard2', -5),
    ('turtle1', 6),
    ('side_control1', 7), 
    ('side_control2', -7,),
    ('back1',  8),
    ('back2', -8),
    ('mount1', 9),
    ('mount2', -9)
]


fig, ax = plt.subplots(figsize=(16, 8))

fontsize=17

ax.set_xlim(0, 9300)
ax.set_ylim(-9.5, 9.5)

ax.set_xticks(list(range(0, 10300, 500)))
ax.set_yticks([-9, -8, -7, -6, -5, -4, -3, -2, -1,  
               0,
               1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yticklabels(['mount', 'back', 'side control', 'turtle', 
                    'half guard', 'closed guard', 'open guard', 'takedown', 
                    '5050 guard', 'POINTS:', 'standing', 
                    'takedown', 'open guard', 'closed guard', 'half guard',
                    'turtle', 'side control', 'back', 'mount'], fontsize=20)
last_x = 0
for decision in gts1:
    x, points, _ = decision
    if last_x + 30 > x:
        y = -0.35
    else:
        y = 0
    ax.text(x, y, f'+{points}', color = '#ff6961', fontsize=fontsize)
    last_x = x

last_x = 0
for decision in gts2:
    x, points, _ = decision
    if last_x + 30 > x:
        y = 0.35
    else:
        y = 0
    ax.text(x, y, f'+{points}', color = '#817bff', fontsize=fontsize)
    last_x = x

for position, y in positions:
    if '2' in position:
        color = '#817bff'
    elif '1' in position:
        color = '#ff6961'
    else:
        color='#f7ff61'
    lines = get_lines(gt_pos, position)
    for line in lines:
        ax.plot(line, np.ones(2) * y, color = color, linewidth=8)
        
last_x = 0
for decision in pts1:
    x, points, _ = decision
    if last_x + 30 > x:
        y = 1.1
    else:
        y = 0.5
    ax.text(x, y, f'+{points}', color = 'red', fontsize=fontsize)
    last_x = x
    
last_x = 0    
for decision in pts2:
    x, points, _ = decision
    if x > 9500:
        break
    if last_x + 30 > x:
        y = -1.1
    else:
        y = -0.5
    ax.text(x, y, f'+{points}', color = 'blue', fontsize=fontsize)
    last_x = x

for position, y in positions:
    if '2' in position:
        color = 'blue'
    elif '1' in position:
        color = 'red'
    else:
        color='orange'
    lines = get_lines(pred_pos, position)
    for line in lines:
        ax.plot(line, np.ones(2) * y, color = color)
        
ax.set_ylabel('Position', fontsize=20)
ax.set_xlabel('Frame', fontsize=20)

fontsize = 16
legend = patches.Rectangle((8400, 3.5), 1600, 5.5, color='white')
ax.add_patch(legend)
ax.text(8500, 3.8, '+2 pred', color = 'blue', fontsize=fontsize)
ax.text(8500, 4.5, '+2 gt', color = '#817bff', fontsize=fontsize)
ax.plot((8500, 8600), (5.3, 5.3), color = '#ff6961', linewidth=10)
ax.plot((8500, 8600), (5.9, 5.9), color = '#817bff', linewidth=10)
ax.plot((8500, 8600), (6.5, 6.5), color = '#f7ff61', linewidth=10)
ax.plot((8500, 8600), (7.1, 7.1), color = 'red')
ax.plot((8500, 8600), (7.7, 7.7), color = 'blue')
ax.plot((8500, 8600), (8.3, 8.3), color = 'orange')
ax.text(8650, 5.2, 'gt 1', fontsize=fontsize)
ax.text(8650, 5.8, 'gt 2', fontsize=fontsize)
ax.text(8650, 6.4, 'gt', fontsize=fontsize)
ax.text(8650, 7, 'pred 1', fontsize=fontsize)
ax.text(8650, 7.6, 'pred 2', fontsize=fontsize)
ax.text(8650, 8.2, 'pred neutral', fontsize=fontsize)
ax.vlines([2500, 5000, 7500], -9.5, 9.5)

fig.tight_layout()
# save the figure
plt.savefig('figures/scoring.png', dpi=300)