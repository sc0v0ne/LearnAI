# Computer Vision Tips

<!-- MarkdownTOC -->

- Data Augmentation for MNIST
- MNIST Image Augmentation Using Tensorflow
  - Step 1: Import the MNIST dataset
  - Step 2: Identify and Plot Baseline Digits Using Matplotlib
  - Step 3:  Understand Image Augmentation and Techniques Relevant To MNIST
    - Overlay Images
  - Step 4: Augment The MNIST Dataset
- Command-line Snippets
- Video/Image manipulation using ffmpeg
  - Check video duration
  - Convert video format
  - Generate dataset from videos
  - Generate a sequence of frames
  - Crop a bounding box of video
- Stack Videos
- Improve Model Performance
- Medical Image Overview
- Medical Image Examples
- References

<!-- /MarkdownTOC -->


## Data Augmentation for MNIST

[Improving Classification accuracy on MNIST using Data Augmentation](https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903?gi=916228e35c66)

We can write a method to shift the images in all four directions by the given order.

We will shift the images to each of the four directions by one pixel and generate four more images from a single image.


----------


# MNIST Image Augmentation Using Tensorflow

[How To Augment the MNIST Dataset Using Tensorflow](https://medium.com/the-data-science-publication/how-to-augment-the-mnist-dataset-using-tensorflow-4fbf113e99a0)

In this tutorial, we will use the `ImageDataGenerator` class in the `tensorflow.keras` python library. 

### Step 1: Import the MNIST dataset

In step 1, we will import the MNIST dataset using the tensorflow library. The imported dataset will be divided into train/test and input/output arrays.

```py
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

### Step 2: Identify and Plot Baseline Digits Using Matplotlib

We plot a subset of the MNIST images to help us understand the augmentation effects on the MNIST dataset. 

To plot a subset of MNIST images, use the following code:

### Step 3:  Understand Image Augmentation and Techniques Relevant To MNIST

The original MNIST dataset contains centered, upright, and size normalized digits. 

Realistically, hand-written digits will seldom meet these criteria in real-world applications. Some digits will be larger, smaller, rotated, or skewed more than others. 

To create a robust digit recognition model, it is in your interest to augment the MNIST dataset and capture these types of behavior. 

We discuss the various types of augmentation techniques we can use to enhance the MNIST digit dataset using the Keras `ImageDataGenerator` class. 

- Rotate
- Shift
- Shear
- Zoom

- Crop (center and random)
- Resize
- Flip (horiz/vert)
- ColorJitter
- Blur
- Greyscale

- Adding Noise
- Saturation
- Cutout
- Filter

_Cutout_ is a simple regularization technique of randomly masking out square regions of input during training which can be used to improve the robustness and overall performance of convolutional neural networks. 

This method can also be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.


_ColorJitter_ is another simple type of image data augmentation where we randomly change the brightness, contrast, and saturation of the image. 

#### Overlay Images

Sometimes, we need to add a background to an existing image for formatting purposes. For instance, by padding a solid color as margins, we can make many images of different sizes become the same shape. Several techniques are relevant here.

### Step 4: Augment The MNIST Dataset

Finally, we can combine all of the previously mentioned transformations to obtain unique digit representations that can now be used to improve digit recognition model performance.


----------


## Command-line Snippets

```py
# Check dimensions of images
import cv2
from pathlib import Path

images = Path('path').rglob('*.png')
for img_path in images:
  img = cv2.imread(img_path)
  print(img_path, img.shape)
```

```py
  # check that count of images and labels are equal

  # count all files
  ls path/to/images | wc -l

  # count of PNG files
  find -name *.png -type | wc -l
```

----------


## Video/Image manipulation using ffmpeg

`ffmpeg` is a very useful utility for CV. 

### Check video duration

```bash
  ffmpeg -i file.mp4 2>&1 | grep “Duration”
```

### Convert video format

```bash
  ffmpeg -i video.mp4 video.avi

  # extract audio only
  ffmpeg -i input.mp4 -vn output.mp3
```

### Generate dataset from videos

```bash
  ffmpeg -ss 00:10:00 -i input_video.mp4 -to 00:02:00 -c copy output.mp4

  # video without audio
  ffmpeg -i input_video.mp4 -an -c:v copy output.mp4
```

where

- -ss starting time
- -i  input video
- -to time interval like 2 minutes.
- -c  output codec
- -an make output without audio.

### Generate a sequence of frames

```py
  # generate images from videos for 20 seconds
  ffmpeg -ss 00:32:15 -t 20 -i videos.ts ~/frames/frame%06d.png
  
  # rescale the images
  ffmpeg -ss 00:10:00 -t 20 -i video.ts -vf scale=iw/2:ih output_path/frame%06d.png
  
  ffmpeg -ss 00:10:00 -t 20 -i video.ts -vf scale=960x540 output_path/frame%06d.png
```

### Crop a bounding box of video

```py
  ffmpeg -i input.mp4 -filter:v "crop=w:h:x:y" output.mp4
```

## Stack Videos

- Horizontally
- Vertically
- 2x2 grid stacking with xstack


----------


## Improve Model Performance

[How to Accelerate Computer Vision Model Inference](https://wallarooai.medium.com/how-to-accelerate-computer-vision-model-inference-98ba449c0f53)



## Medical Image Overview

[AI in Medical Diagnosis — Dealing with Medical Datasets](https://towardsdatascience.com/ai-in-medical-diagnosis-dealing-with-medical-datasets-b746e8bda9e5)

[Medical Image Dataloaders in TensorFlow 2.x](https://towardsdatascience.com/medical-image-dataloaders-in-tensorflow-2-x-ee5327a4398f)

[Medical Image Pre-Processing with Python](https://towardsdatascience.com/medical-image-pre-processing-with-python-d07694852606)

[Computer Vision Feature Extraction 101 on Medical Images](https://towardsdatascience.com/computer-vision-feature-extraction-101-on-medical-images-part-1-edge-detection-sharpening-42ab8ef0a7cd)


## Medical Image Examples

[Chest X-ray & Pneumonia: Deep Learning with TensorFlow](https://towardsdatascience.com/chest-x-ray-pneumonia-deep-learning-with-tensorflow-a58a9e6ade70)

[Deep Learning in Medicine: Identifying Pneumonia from a Chest-X-Ray](https://medium.com/@atulranjan975/deep-learning-in-medicine-identifying-pneumonia-from-a-chest-x-ray-6d1aabe68b01)

[Artificial intelligence for lung disease detection using chest CT scan images](https://medium.com/@atashnezhad1/artificial-intelligence-for-lung-disease-detection-using-chest-ct-scan-images-47e8ff279ce0)


[Compositional Breast Imaging with Artificial Intelligence to Improve Cancer Detection](https://towardsdatascience.com/compositional-breast-imaging-with-artificial-intelligence-to-improve-cancer-detection-c5f8f38f2266?source=rss----7f60cf5620c9---4)



## References

[Achieving 95.42% Accuracy on Fashion-Mnist Dataset](https://secantzhang.github.io/blog/deep-learning-fashion-mnist)

