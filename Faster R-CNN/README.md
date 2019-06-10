# Faster R-CNN
A minimal PyTorch implementation of Faster R-CNN, with support for training, predicting and evaluation. The dataset is Pascal VOC 2012. Concept of transfer learning is applied to implement Resnet-50, DenseNet-121, Vgg-16 with batch normalization and MobileNet_v2.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/LegendStatus/Object_Detector

The version of pytorch is 1.1.0 or above. You can update the pytorch by running:
    $ pip install --user --upgrade torch torchvision

##### Set up dataset path

The dataset is stored in ieng6 cluster. You can access the dataset at '/datasets/ee285f-public/PascalVOC2012' on ieng6 cluster. 

## Visulaize Data

The dataset visulization is contained in the file:

```
$ Visualize Dataset.ipynb
```

## Train

The training processes for each model are in the file:
```
$ Faster_RCNN-Desnet121.ipynb
$ Faster_RCNN-Resnet.ipynb
$ Faster_RCNN-mobilenet.ipynb
$ Faster_RCNN-vgg16.ipynb
```
You can simply rerun the notebook.

The training and saving model functions are contained in the file:

```
$ nntools.py
```

## Evaluation

This would compute the AP of each class and overall mAP of the validation dataset. The evaluations of each model are contained in the same notebook as training process. 

```
$ Faster_RCNN-Desnet121.ipynb
$ Faster_RCNN-Resnet.ipynb
$ Faster_RCNN-mobilenet.ipynb
$ Faster_RCNN-vgg16.ipynb
```

The metric function is written in the file:

```
$ evaluation_voc.py
```

The plot of AP of each class and overall mAP is in the file:

```
$ Evaluation.ipynb
```


## Prediction and Demo

The demo for each model is in the file:
```
$ demo_Faster_RCNN_DenseNet121.ipynb
$ demo_Faster_RCNN_ResNet50.ipynb
$ demo_Faster_RCNN_VGG16.ipynb
$ demo_Faster_RCNN_MobileNet_v2.ipynb
```
Some outputs and results are contained in the Result folder.



