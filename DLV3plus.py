import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from _function.func_data_generator import data_generator
from _function.DLV3plus_model import DeepLabV3Plus

################################## GET TRAIN DATA #######################################

train_images_0 = []
for i in sorted(os.listdir('leftImg8bit\\train')):
    folder_path = 'leftImg8bit\\train\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train\\' + i + '\\' + j
        train_images_0.append(img_path)

train_images_light_1 = []
for i in sorted(os.listdir('leftImg8bit\\train_light_1')):
    folder_path = 'leftImg8bit\\train_light_1\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train_light_1\\' + i + '\\' + j
        train_images_light_1.append(img_path)

train_images_light_2 = []
for i in sorted(os.listdir('leftImg8bit\\train_light_2')):
    folder_path = 'leftImg8bit\\train_light_2\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train_light_2\\' + i + '\\' + j
        train_images_light_2.append(img_path)

train_images_night_1 = []
for i in sorted(os.listdir('leftImg8bit\\train_night_1')):
    folder_path = 'leftImg8bit\\train_night_1\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train_night_1\\' + i + '\\' + j
        train_images_night_1.append(img_path)

train_images_night_2 = []
for i in sorted(os.listdir('leftImg8bit\\train_night_2')):
    folder_path = 'leftImg8bit\\train_night_2\\' + i
    files = sorted(os.listdir(folder_path))
    for j in files:
        img_path = 'leftImg8bit\\train_night_2\\' + i + '\\' + j
        train_images_night_2.append(img_path)

train_masks_0 = []
for i in sorted(os.listdir('mask\\train')):
    train_masks_0.append('mask\\train\\' + i)
train_mask_light_1 = train_masks_0.copy()
train_mask_light_2 = train_masks_0.copy()
train_mask_night_1 = train_masks_0.copy()
train_mask_night_2 = train_masks_0.copy()

############################################### GET VALID DATA #################################################

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

################################################ PREPARE DATA ###################################################

train_images = train_images_0 + train_images_light_1 + train_images_light_2 + train_images_night_1 + train_images_night_2
train_masks = train_masks_0 + train_mask_light_1 + train_mask_light_2 + train_mask_night_1 + train_mask_night_2

#USING 2 LINES BELOW IF YOU DONT WANT TO USE AUGMENTATION DATA
#train_images = train_images_0 
#train_masks = train_masks_0 

train_dataset = data_generator(train_images, train_masks, batch_size=4)
val_dataset = data_generator(val_images, val_masks, batch_size=2)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

############################################### TRAINING MODEL #######################################################

CLASSES = 6
SHAPE = (256, 256, 3)

model = DeepLabV3Plus(num_class=CLASSES, input_shape=SHAPE)
#model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss)

history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, verbose=1) # using callbacks=callback if you want to use

model.save('model_DLV3plus.h5')

print('SAVED MODEL!')

############################################### PLOT TRAINING RESULT #####################################################

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()