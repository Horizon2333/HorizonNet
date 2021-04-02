# HorizonNet

Simple code for image classification based on Pytorch

##  Support models

|  Series   | Models                                                       |
| :-------: | :----------------------------------------------------------- |
|  AlexNet  | AlexNet (1 branch), AlexNetv1 (2 branches, the same as original paper, simulated by forward function in Pytorch) |
|    VGG    | VGG11, VGG11LRN, VGG13, VGG16, VGG16conv1, VGG19             |
| GoogLeNet | GoogLeNet, GoogLeNet_aux (with auxiliary branch)             |
|  ResNet   | ResNet18, ResNet34, ResNet50, ResNet101, ResNet152           |
| DenseNet  | DenseNet121, DenseNet169, DenseNet201, DenseNet264           |
| SimpleNet | SimpleNet96(a simple self-defined convolution neural network) |

## Project structure
```
$videoqa_dataset_visualization
    |──checkpoints              # model checkpoints
        |──xxxx.pth
    |──logs                     # console output
        |──train.log            # train.py console output
        |──model.log            # print_model.py console output
    |──models                   # support models
        |──AlexNet.py
        |──VGG.py
        |──GoogLeNet.py
        |──ResNet.py
        |──DenseNet.py
        |──SimpleNet.py
        |──LoadModels.py
    |──pretrained               # save torchvision pretrained model here to save disk space in C:\
        |──xxxx.pth
    |──runs                     # tensorboard log
        |──xxxx
    |──logger.py                # produce log file
    |──Horizon.py               # print me
    |──divide_train_val.py      # divide training and validation set
    |──print_model.py           # print model information: layers, params, flops
    |──train.py                 # train models
    |──result.jpg               # save result chart
    |──requirements.txt
    |──readme.md
```

## Dataset structure

```
$your_dataset_path
    |──train
        |──class1
            |──xxxx.jpg
            |──...
        |──class2
            |──xxxx.jpg
            |──...
        |──...
        |──classN
            |──xxxx.jpg
            |──...
    |──val
        |──class1
            |──xxxx.jpg
            |──...
        |──class2
            |──xxxx.jpg
            |──...
        |──...
        |──classN
            |──xxxx.jpg
            |──...
```

## Install

1. Clone the project
```shell
git clone https://github.com/Horizon2333/HorizonNet
cd HorizonNet
```
2. Install dependencies
```shell
pip install -r requirements.txt
```

## Usage

To train classification models:

```shell
python train.py --model ResNet18 --dataset_path [your_dataset_path]
```



