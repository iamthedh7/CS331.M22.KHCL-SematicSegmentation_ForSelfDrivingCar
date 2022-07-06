<!-- Title -->
<h1 align="center"><b>CS331.M22.KHCL</b></h1>
<h1 align="center"><b>SEMATIC SEGMENTATION FOR SELF DRIVING CAR</b></h1>


Using: UNET and DEEPLABv3+ in Tensorflow framework

---------------------------------------------------------------------------------------------

**# Step 0:** clone this repository

**# Step 1:** install CUDA and cudnn (recommend) if you're using Nvidia GPU.

**# Step 2:** set up an Anaconda Environment.

**# Step 3:** go to https://drive.google.com/drive/folders/1iKLyJCLocV90glBT67hvCnZJRsFBJTSq?usp=sharing to download raw data (2 folders - we use CityScape dataset)

**# Step 3:** cd to your repository's savepath, then activate the Environment and run this command line:

           python Main.py 

# or:  <if you want to save your Result>     
    
           python Main.py 1

**# Step 4:** select your custom test image or some image in folder Testing. The result image will be shown for 10s.

           
_***Note:_ Please install the necessary libraries into the Environment by yourself when the terminal warning "missing libraries". Detecting in a video and realtime will be update soon.

           
------------------------------------------------------------------------------------------------

Note: Our model is using 6 classes: Background, Car0 (Carhead), Road, Car, Cycle, Human

<h1 align="center"><b>--------</b></h1>

<!-- Footer -->
<p align='center'>Copyright © 2022 Duong Hai Nguyen, Tuan Nam Trinh, Hoai Nam Nguyen, Thanh Trung Nguyen</p>
