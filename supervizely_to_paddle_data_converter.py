import json
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def convert_bbox_to_four_points(bbox):
    # Unpack the bounding box
    top_left, bottom_right = bbox

    # Create the additional points
    top_right = [bottom_right[0], top_left[1]]
    bottom_left = [top_left[0], bottom_right[1]]

    # Create the 4-point bounding box
    bbox_4_points = [ top_left, top_right, bottom_right, bottom_left ]

    return bbox_4_points

def convert_to_paddle_format(root_directory, output_directory):
    # Make sure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    all_tags = set()

    # Iterate over the 'val', 'train', 'test' subdirectories
    for dataset in ['val', 'train', 'test']:
        paddle_data = []

        # Path to the annotations subdirectory
        ann_directory = Path(root_directory + '/' + dataset + '/' + 'ann').as_posix()

        # Iterate over all files in the given directory
        for filename in tqdm(os.listdir(ann_directory), total=len(os.listdir(ann_directory))):
            if filename.endswith('.json'):  # Process only JSON files
                with open(os.path.join(ann_directory, filename), 'r') as f:
                    data = json.load(f)

                image_annotations = []
                for obj in data['objects']:
                    paddle_obj = {}

                    # Extract bounding box coordinates
                    bbox = np.array(obj['points']['exterior']).tolist()

                    # Extract class type
                    class_label_type = obj['tags'][0]['value']
                    all_tags.add(obj['classTitle'])

                    paddle_obj['transcription'] = class_label_type
                    paddle_obj['points'] = convert_bbox_to_four_points(bbox)

                    image_annotations.append(paddle_obj)

                # Construct a line in the paddle format
                line = f"{filename.replace('.json', '')}\t{json.dumps(image_annotations)}\n"
                paddle_data.append(line)

        # Write to the corresponding output file
        with open(os.path.join(output_directory, f'{dataset}_data.txt'), 'w') as f:
            f.writelines(paddle_data)



def draw_annotations(image_directory, annotations_file):
    # Open the annotations file
    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Split the line into image filename and annotation
        filename, annotation = line.strip().split('\t')
        
        # Load the image
        img = cv2.imread(Path(image_directory + '\\' + filename).as_posix())

        # Parse the JSON annotation
        annotations = json.loads(annotation)

        # Draw each polygon
        for ann in annotations:
            points = np.array(ann['points'], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img, [points], True, (0, 255, 0), 2)

        # Show the image
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    convert_to_paddle_format("data/train_test_val_split", "data/data_in_paddle_format")
    draw_annotations("data/data_in_paddle_format/train", "data/data_in_paddle_format/train_data.txt")

