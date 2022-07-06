import cv2
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
from _function.func_read_image import read_image

path = 'model_DLV3plus.h5' # insert your model's path here
model_loaded = keras.models.load_model(path)

st = time.time()
for i in os.listdir('leftImg8bit\\test\\bonn'):
    path = 'leftImg8bit\\test\\bonn\\' + i
    image_tensor = read_image(path)
    predictions = model_loaded.predict(np.expand_dims((image_tensor / 255), axis=0))  #(1,256,256,6)
    predictions = tf.squeeze(predictions, axis=0) #(256,256,6)
    predictions = tf.math.argmax(predictions, axis=2)

    rgb = []
    content = []
    for i in predictions.numpy().ravel():
        temp = [i] * 3
        if (i == 0.0):
            temp = [180, 180, 180]
        elif (i == 1.0):
            temp = [180, 0, 0]
        elif (i == 2.0):
            temp = [0, 180, 0]
            content.append('Road')
        elif (i == 3.0):
            temp = [0, 0, 180]
            content.append('Human')
        elif (i == 4.0):
            temp = [180, 180, 0]
            content.append('Car')
        elif (i == 5.0):
            temp = [0, 180, 180]
            content.append('Cycle')
        rgb.append(temp)
    rgb = np.array(rgb)
    rgb = rgb.reshape(256, 256, 3).astype(np.uint8)

    added_image = cv2.addWeighted(image_tensor.numpy().astype(np.uint8), 1, rgb, 0.5, -2)
    added_image = cv2.resize(added_image, (int(2048/3), int(1024/3)))

    temp = 25
    if ('Road' in content):
        cv2.putText(added_image, 'Road', (10, temp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)
        temp = 50
    if ('Human' in content):
        cv2.putText(added_image, 'Human', (10, temp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 2)
        temp = 75
    if ('Car' in content):
        cv2.putText(added_image, 'Car', (10, temp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 2)
        temp = 100
    if ('Cycle' in content):
        cv2.putText(added_image, 'Cycle', (10, temp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 2)

    cv2.imshow('demo', added_image)
    time.sleep(0.5)
    c = cv2.waitKey(1)
    if c == 27:
        break 

cv2.destroyAllWindows()
end = time.time()

print('# Time:', end-st, 's')