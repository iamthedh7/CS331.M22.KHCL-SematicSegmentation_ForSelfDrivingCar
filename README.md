<!-- Title -->
<h2 align="center"><b>CS331.M22.KHCL</b></h2>
<h1 align="center"><b>SEMATIC SEGMENTATION FOR SELF DRIVING CAR</b></h1>
<h2 align="center"><b>~~~~~~~</b></h2>

---------------------------------------------------------------------------------------------

**Using:** UNET and DEEPLABv3+ in Tensorflow framework

**Dataset:** we use CityScape dataset (https://www.cityscapes-dataset.com/downloads/)

---------------------------------------------------------------------------------------------

**Note:** 

1. Your Anaconda Environment should be installed 'tensorflow-gpu' if you're using Nvidia GPU.

2. Our model is using 6 classes: Background, Car0 (Carhead), Road, Car, Cycle, Human.

---------------------------------------------------------------------------------------------

**# Step 0:** clone this repository and rename to CV

**# Step 1:** go to https://drive.google.com/drive/folders/1iKLyJCLocV90glBT67hvCnZJRsFBJTSq?usp=sharing to download 2 folders.

**# Step 2:** extract compressed file in _Step 2_ and put into CV (leftImg8bit: images, gtFine: masks).

**# Step 3:** cd to CV folder and run: 

- This cmd line uses to augmentation our raw data. Here, we change the image's brightness up (+30, +60) and down (-20, -40):

      <cmd> 
      python Augmentation.py

- Then we will train our model. When run these sequencely, we have _model_Unet.h5_ & _model_DLV3plus.h5_:

      <cmd>
      python UNET_model.py
      python DLV3plus.py

- We will calculate IoU score in both models. It uses both train-raw-data & valid-raw-data to calculate, instead of a separate test set:
      
      <cmd>
      python Evaluate.py

- If we want to see how the results are predicted with your model, open Visual Studio Code and run: **_'Single_pred.ipynb'_**.
      
- Run this cmd line to show many result images like a video. You can edit a little bit to use for a video as input:
      
      <cmd>
      python Multi_pred.py

------------------------------------------------------------------------------------------------
<!-- Footer -->
<p align='center'>Copyright © 2022 Duong Hai Nguyen, Tuan Nam Trinh, Hoai Nam Nguyen</p>
