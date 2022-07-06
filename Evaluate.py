import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.metrics import MeanIoU
from _function.func_read_image import read_image

########################################### GET DATA ##########################################

train_images_0 = []
for i in sorted(os.listdir('leftImg8bit\\train')):
    folder_path = 'leftImg8bit\\train\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train\\' + i + '\\' + j
        train_images_0.append(img_path)

train_masks_0 = []
for i in sorted(os.listdir('mask\\train')):
    train_masks_0.append('mask\\train\\' + i)

val_images = []
for i in sorted(os.listdir('leftImg8bit\\val')):
    folder_path = 'leftImg8bit\\val\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\val\\' + i + '\\' + j
        val_images.append(img_path)

val_masks = []
for i in sorted(os.listdir('mask\\val')):
    val_masks.append('mask\\val\\' + i)

all_imgs = train_images_0 + val_images
all_masks = train_masks_0 + val_masks

############################################## IMPORT MODEL ############################################

path1 = 'model_UNET.h5' # path to UNET model
path2 = 'model_DLV3plus.h5' # path to DLV3plus model

model_UNET = keras.models.load_model(path1)
model_DLV3plus = keras.models.load_model(path2)

############################################## CALCULATE IoU SCORE ##############################################

sum_IoU_0 = 0
sum_IoU_1 = 0
sum_IoU_2 = 0
sum_IoU_3 = 0

CLASSES = 6
m = MeanIoU(num_classes=CLASSES)

for i in range(len(all_imgs)):
    if (i % 100 == 0) or (i % len(all_imgs) == 0):
        os.system('cls')
        precent = int((i+1) / len(all_imgs))
        print('---' + str(precent * 100) + ' % ---')

    image_tensor = read_image(all_imgs[i])
    mask_tensor = read_image(all_masks[i], mask=True)
    mask_tensor = tf.squeeze(mask_tensor, axis=2)

    predictions_0 = model_DLV3plus.predict(np.expand_dims((image_tensor), axis=0))  #(1,256,256,6)
    predictions_0 = tf.squeeze(predictions_0, axis=0) #(256,256,6)
    predictions_0 = tf.math.argmax(predictions_0, axis=2) 
    predictions_0 = tf.cast(predictions_0, dtype=tf.float32)
    m.update_state(mask_tensor, predictions_0)
    sum_IoU_0 += m.result().numpy()
    m.reset_state()

    predictions_2 = model_UNET.predict(np.expand_dims((image_tensor), axis=0))  #(1,256,256,6)
    predictions_2 = tf.squeeze(predictions_2, axis=0) #(256,256,6)
    predictions_2 = tf.math.argmax(predictions_2, axis=2) 
    predictions_2 = tf.cast(predictions_2, dtype=tf.float32)
    m.update_state(mask_tensor, predictions_2)
    sum_IoU_2 += m.result().numpy()
    m.reset_state()

print('# Average IoU score of DeepLab v3+ augmentation model: ', end='')
print(round(sum_IoU_0/len(all_imgs) * 100, 4), '%')
print('# Average IoU score of UNET augmentation model: ', end='')
print(round(sum_IoU_2/len(all_imgs) * 100, 4), '%')