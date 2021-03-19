# HorizonNet

Simple code for image classification based on Pytorch

# Support models

- AlexNet (1 branch) , AlexNetv1 (2 branches, the same as original paper, simulated by forward function in Pytorch)
- VGG11, VGG11LRN, VGG13, VGG16, VGG16conv1, VGG19
- GoogLeNet, GoogLeNet_aux (with auxiliary branch)
- ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- DenseNet121, DenseNet169, DenseNet201, DenseNet264

# train

For training classification models:

```shell
python train.py --model ResNet18 --dataset_path [your_dataset_path]
```



