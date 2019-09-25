# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import os
import numpy as np
import time



import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


model_path = './resnet50_csv_12_inference.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')


labels_to_names = {0: 'Biker', 1: 'Car', 2: 'Bus', 3: 'Cart', 4: 'Skater', 5: 'Pedestrian'}

sdd_images = os.listdir('./examples_UC3M')
print(sdd_images)


def run_detection_image(filepath):
    image = read_image_bgr(filepath)
    #     print(image.shape)
    #     print(image)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    file, ext = os.path.splitext(filepath)
    image_name = file.split('/')[-1] + ext
    output_path = '/home/jetsontx2/mahmoud/aerial_pedestrian_detection/output_UC3M/'+ image_name
    print(output_path)
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, draw_conv)

base_path = './examples_UC3M'

for image in sdd_images:
    if 'jpg' in image:
    	print(image)
    	run_detection_image(os.path.join(base_path,image))
