import sys
sys.path.append('/usr/local/cv3.5/lib/python3.5/dist-packages')

import os
import time
import numpy as np
import cv2

import coco
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.print()


# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def norm_color(color):
    norms = []
    for i in range(3):
        norms.append(color[i] / 255.0)
    return (norms[0], norms[1], norms[2])


# Electronic fence
# fence_pt1 = (10, 450)
# fence_pt2 = (1269, 600)
# fence_pt1 = (10, 470)
# fence_pt2 = (1089, 650)
fence_pt1 = (0, 0)
fence_pt2 = (0, 0)
fence_color_normal = (226, 137, 59)
fence_color_warm = (66, 194, 244)

# To detect
detected_class_names = ['person']
detected_color = (0, 0, 200)
detected_threshold = 0.95
in_region_count = 20


fence_color_normal = norm_color(fence_color_normal)
fence_color_warm = norm_color(fence_color_warm)
detected_color = norm_color(detected_color)


def main(src, dst):
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print('Failed to open {}'.format(src))
        return

    # Image size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Fence range
    fence_x1 = fence_pt1[0]
    fence_x2 = fence_pt2[0]
    fence_y1 = fence_pt1[1]
    fence_y2 = fence_pt2[1]
    # Fence mask
    fence_mask = np.zeros((height, width)).astype(np.uint8)
    fence_mask[fence_y1:fence_y2, fence_x1:fence_x2] = 1

    # Video writer
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', '3')
    out = cv2.VideoWriter(dst, fourcc, 20.0, (width, height))

    total_elapsed_time = 0
    total_frame_count = 0
    in_region = False
    in_region_counter = 0
    while True:
        success, image = cap.read()
        if not success:
            break

        # timestamp before detection
        time_before = time.time()

        # Run detection
        results = model.detect([image], verbose=0)

        # Calculate and show FPS
        elapsed_time = time.time() - time_before
        total_elapsed_time += elapsed_time
        total_frame_count += 1
        fps = total_frame_count / total_elapsed_time
        print('FPS: {}'.format(fps))

        # Visualize results
        r = results[0]
        fence_color = fence_color_warm if in_region else fence_color_normal
        _in_region, masked_image = visualize.fence_display(
            image, r['masks'], r['class_ids'], class_names, r['scores'],
            detected_class_names, detected_color, detected_threshold,
            fence_mask, fence_color)

        # Update region status if necessary
        if _in_region:
            in_region_counter = in_region_count
            in_region = True
        elif in_region:
            in_region_counter -= 1
            if in_region_counter == 0:
              in_region = False

        # Write the result image to output video
        out.write(masked_image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python demo.py src dst')
        sys.exit(-1)

    src = sys.argv[1]
    dst = sys.argv[2]
    print('Input video: {}'.format(src))
    print('Output video: {}'.format(dst))
    main(src, dst)
