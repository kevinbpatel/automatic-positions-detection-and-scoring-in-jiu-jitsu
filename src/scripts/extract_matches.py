import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, default='outputs', help='Root of the output files.')
    parser.add_argument('--flip-ids', action='store_true', help='Flip the ids of the players if necessary')
    return parser.parse_args()

def get_positions(output_dir, suffix = '', flip=False):
    predictions_path = f'{output_dir}/predictions/predictions{suffix}.pickle'
    if not os.path.isfile(predictions_path):
        return []
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    
    matches = dict()
    for frame, results in predictions.items():
        matched = dict()
        p1, p2 = (None, None)
        id1, id2 = (None, None)
        ioa = 0
        for i, result in enumerate(results):
            if result['track_id'] == 0 and p1 is None:
                p1 = result['keypoints']
                id1 = i
            if result['track_id'] == 1 and p2 is None:
                p2 = result['keypoints']
                id2 = i
            if result['ioa'] > ioa:
                    ioa = result['ioa']
        if id1 is not None:
            key = 1 if not flip else 2
            matched[key] = {'keypoints': p1,
                          'id': id1}

        if id2 is not None:
            key = 2 if not flip else 1
            matched[key] = {'keypoints': p2,
                          'id': id2}
        matched['ioa'] = ioa
        matches[frame] = matched
    return matches


def main():
    args = parse_args()
    matches = get_positions(args.output_dir, flip=args.flip_ids)

    with open(f'{args.output_dir}/predictions/predictions_matched.pickle', 'wb') as f:
        pickle.dump(matches, f)

if __name__ == '__main__':
    main()

        