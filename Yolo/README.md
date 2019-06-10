# YOLO_v1
A minimal PyTorch implementation of YOLOv1, with support for training, predicting and evaluation. The dataset is Pascal VOC 2012. Concept of transfer learning is applied to implement Resnet-50 and Vgg-16 with batch normalization.

## Installation
##### Clone and install requirements
    $ git lfs track "*.pth"
    $ git clone https://github.com/LegendStatus/Object_Detector
    $ cd Object_Detector/yolo
    $ sudo pip3 install -r requirements.txt

##### Set up dataset path

Change root directory of training image to your own dataset in line 30 of ***train.py***, ***prediction.ipynb*** and ***evaluation.ipynb***. Default directory is './data/JPEGImages/'. If it is running on DSMLP, you can do

    $ mkdir data
    $ cd data
    $ ln -s /datasets/ee285f-public/PascalVOC2012/

##### Converting xml to txt

This would generate the list of image that are needed for training and evaluation of PascalVOC2012. Root directory in ***xml_2_txt.py*** also is required to change

```
$ python xml_2_txt.py --mode 'val'or'train'
```

## Train

```
$ python train.py [--epochs EPOCHS] 
                  [--batch_size BATCH_SIZE]
                  [--learning_rate Initial learning rate for SGD]
                  [--img_size IMG_SIZE]
                  [--net 'resnet50'or'vgg16_bn']
```

## Demo

##### Prediction

This would randomly pick 5 images from validation dataset and predict based on the network that is selected(Resnet-50 or Vgg-16_bn)

```
$ prediction.ipynb
```

##### Evaluation

This would compute the AP of each class and overall mAP of the validation dataset.

```
$ evaluation.ipynb
```
