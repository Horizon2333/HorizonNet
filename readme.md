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
    |──checkpoints              # result image
        |──xxxx.pth
    |──logs
        |──train.txt
        |──model.txt
    |──models
        |──AlexNet.py
        |──VGG.py
        |──GoogLeNet.py
        |──ResNet.py
        |──DenseNet.py
        |──SimpleNet.py
        |──LoadModels.py
    |──pretrained
        |──xxxx.pth
    |──runs
        |──xxxx
    |──logger.py
    |──Horizon.py
    |──divide_train_val.py
    |──print_model.py
    |──train.py
    |──result.jpg             
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



