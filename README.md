<!-- Title -->
<h2 align="center"><b>CS331.M22.KHCL</b></h2>
<h1 align="center"><b>SEMATIC SEGMENTATION FOR SELF DRIVING CAR</b></h1>
<h2 align="center"><b>~~~~~~~</b></h2>

  Using: UNET and DEEPLABv3+ in Tensorflow framework

  Dataset: we use CityScape dataset (https://www.cityscapes-dataset.com/downloads/)

---------------------------------------------------------------------------------------------

**Note:** your Anaconda Environment should be installed 'tensorflow-gpu' if you're using Nvidia GPU.

**# Step 0:** clone this repository and rename to CV

**# Step 1:** go to https://drive.google.com/drive/folders/1iKLyJCLocV90glBT67hvCnZJRsFBJTSq?usp=sharing to download 2 folders.
(leftImg8bit: images, gtFine: masks).

**# Step 2:** extract compressed file in _Step 2_ and put into CV.

**# Step 3:** cd to CV folder and run: 

                                                       python <filename>

+ Augmentation.py: 

      this file uses to augmentation our raw data. Here, we change the image's brightness up (+30, +60) and down (-20, -40).

+ UNET_model.py & DLV3plus.py: 

      these files are our model training files. When run these files sequencely, we have _model_Unet.h5_ & _model_DLV3plus.h5_.

+ Evaluate.py: 

      this file will calculate IoU score in both models. It uses both train-raw-data & valid-raw-data to calculate, instead of separate test set.

+ Single_pred.ipynb: 
      
      when we run this file, we can see how the results are predicted with your model.

+ Multi_pred.py: 

      run this file to show many result images like a video. You can edit a little bit to use for a video as input.
           
------------------------------------------------------------------------------------------------

Note: Our model is using 6 classes: Background, Car0 (Carhead), Road, Car, Cycle, Human.

<h1 align="center"><b>--------</b></h1>

<!-- Footer -->
<p align='center'>Copyright © 2022 Duong Hai Nguyen, Tuan Nam Trinh, Hoai Nam Nguyen, Thanh Trung Nguyen</p>
