# HorizonNet

Simple code for image classification based on Pytorch

## Features
- Beautiful console output with tables and progress bar.
- Automatically print model layers and parameters.
- Automatically draw a chart for loss and accuracy.
- Load checkpoints and continue to train.
- Frozen convolution layers to fine-tune fully-connected layers.
- Compute and print confusion matrix.

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
    |──results                  # material pictures
        |──train1.png
        |──train2.png
        |──train3.png
        |──train4.png
    |──logger.py                # produce log file
    |──Horizon.py               # print me
    |──divide_train_val.py      # divide training and validation set
    |──printmodel.py            # print model information: layers, params, flops
    |──train.py                 # train models
    |──result.jpg               # save result chart
    |──requirements.txt
    |──readme.md
```

## Prepare dataset

Dataset structure should like:

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

If you have a dataset like:
```
$your_dataset_path
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
You can run
```shell
python divide_train_val.py --initial_dataset {your dataset path} --output {output path} train_percentage 70
```
```70``` means take 70% of whole dataset into training set, 30% into validation set.

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

To print models:
```shell
python printmodel.py --model ResNet18 --dataset_path [your_dataset_path]
```

## Results

Training progress:

![train1](https://github.com/Horizon2333/HorizonNet/blob/main/results/train1.png)
![train2](https://github.com/Horizon2333/HorizonNet/blob/main/results/train2.png)
![train3](https://github.com/Horizon2333/HorizonNet/blob/main/results/train3.png)
![train4](https://github.com/Horizon2333/HorizonNet/blob/main/results/train4.png)


