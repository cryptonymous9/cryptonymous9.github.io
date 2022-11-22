---
layout: post
title:  "Image recognition"
date:   2022-07-05 19:45:31 +0530
categories: science
author: "Nidhin"
usemathjax: true
---

# A study on GoogLeNet, ResNet and Preact-ResNet on Cifar-10 classification

In this notebook, I'll be implementing some of architectures that we studied in the last few lectures for cifar-10 classification. Specifically,  I'll be comparing 3 methods: 

[1] **GoogLeNet:** Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[2] **ResNet:** He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[3] **Preact-ResNet (improved resnet):** He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.


## Table of Contents:

### - 1. Dataset
        1.1 Data analysis (Abridged) ...............
        1.2 Data Visualization .....................

### - 2.  Image Recognition Architectures
        2.1 GoogLeNet .....................
        2.2 ResNet-18 ...............................
        2.3 PreAct-R18 ......................
        
### - 3. Inference and Evaluation

### - 4. Discussion

### - References

**Reproducibility:** This notebook was ran on the following configuration:
- Python version used is 3.7
- All the cpu-intensive processing is done over `Intel Xeon(R)` chipset.
- All the cuda-processing (including training and inference) has been done over `NVIDIA RTX-3090`


```python
import os
import argparse
import pickle

import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
```

# 1.  CIFAR-10 Dataset

While the actual implementation of above mentioned models were mostly trained on Imagenet dataset (~1.2 million images). Due to computational time constraints, I plan on using a lighter dataset like the CIFAR-10 dataset, which is a classical computer-vision dataset for object recognition case study. 

Below function will download CIFAR10 dataset from the official [PyTorch datasets](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)


When downloaded, CIFAR-10 data will consist of thousands of RGB images. The preprocessing of CIFAR-10 will need the following steps:
- Obtaining the mean and standard deviation of the data so it can be normalized. Each image in the dataset is made up of three channels RGB (red, green and blue). Therefore, means and standard deviations for each of the color channels needs to calculated independently. The mean and standard deviation values that is used to normalize are standard and publicly available. I just straightaway define them.


I apply the following set of specific augmentations to our CIFAR-10 dataset:

 - `RandomHorizontalFlip` - This, with a probability of 0.5 as specified, flips the image horizontally.
 - `RandomCrop` - takes a random 32x32 square crop of the image.
 - `ToTensor` - converts image from a PIL image into a PyTorch tensor.
 - `Normalize` - this subtracts the image pixels channel-wise with its mean and divides by the given standard deviation.


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing data..')

# Data Augmentations ---------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# ------------------------------------------------------------------------------


# Downloading train-set of CIFAR-10 (with a batch-size of 128)
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)


# Downloading test-set CIFAR-10 (with a batch-size of 128)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

    ==> Preparing data..
    Files already downloaded and verified
    Files already downloaded and verified
    


```python
print(f'Number of training examples: {len(trainset)}')
print(f'Number of testing examples: {len(testset)}')
```

    Number of training examples: 50000
    Number of testing examples: 10000
    

## 1.1 Data analysis (Abridged) 

I have already covered exploratoary data analysis in my previous notebook, so I won't be going all over it again. Instead, here's some of the key data-related statistics and its visualization. 


```python
print("\nNumber of classes:", len(trainset.classes))
print("Classes:", trainset.classes)
```

    
    Number of classes: 10
    Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    


```python
print("Distribution of classes (Test)")
plt.hist(testset.targets)
plt.xlabel("Classes")
plt.ylabel("No. of samples")
plt.show()
```

    Distribution of classes (Test)
    


    
![png](/assets/blog/recognition/output_10_1.png)
    



```python
print("Distribution of classes (Train)")
plt.hist(trainset.targets)
plt.xlabel("Classes")
plt.ylabel("No. of samples")
plt.show()
```

    Distribution of classes (Train)
    


    
![png](/assets/blog/recognition/output_11_1.png)
    


CIFAR10 is a highly balanced dataset with equal distribution in each class as can be seen from above histograms.

## 1.2 Visualization

Here, I create a function to plot some images in our dataset to see what they actually look like. Note that by default PyTorch handles images that are arranged `[channel, height, width]`, but matplotlib expects images to be `[height, width, channel]`, hence we need to permute our images before plotting them.


```python
from torchvision.utils import make_grid
import ipyplot

def show(imgs):
    plt.rcParams["figure.figsize"] = (10,20)
    plt.imshow(imgs.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
```


```python
labels_set = trainset.targets[:30]
textual_labels = [trainset.classes[labels_set[i]] for i in range(30)]
ipyplot.plot_class_representations(trainset.data[:30],textual_labels)
```



<style>
    #ipyplot-html-viewer-toggle-XgcCUJT2eggQmCu6kuBhZ9 {
        position: absolute;
        top: -9999px;
        left: -9999px;
        visibility: hidden;
    }

    #ipyplot-html-viewer-label-XgcCUJT2eggQmCu6kuBhZ9 { 
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: blue;
        text-decoration: underline;
    }

    #ipyplot-html-viewer-textarea-XgcCUJT2eggQmCu6kuBhZ9 {
        background: lightgrey;
        width: 100%;
        height: 0px;
        display: none;
    }

    #ipyplot-html-viewer-toggle-XgcCUJT2eggQmCu6kuBhZ9:checked ~ #ipyplot-html-viewer-textarea-XgcCUJT2eggQmCu6kuBhZ9 {
        height: 200px;
        display: block;
    }

    #ipyplot-html-viewer-toggle-XgcCUJT2eggQmCu6kuBhZ9:checked + #ipyplot-html-viewer-label-XgcCUJT2eggQmCu6kuBhZ9:after {
        content: "hide html";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: white;
        cursor: pointer;
        color: blue;
        text-decoration: underline;
    }
</style>
<div>

<div id="ipyplot-imgs-container-div-HsrYJwLSqbdaJxxbQi8oGe">
<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-j6yQagdiWEbv4JbtWDVnG9" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">airplane</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-j6yQagdiWEbv4JbtWDVnG9">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-exQ9mRosZ3hL3QjKo97cdd" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">automobile</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-exQ9mRosZ3hL3QjKo97cdd">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-fk8DbsRfpheMQDT3CAtB5Z" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">bird</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-fk8DbsRfpheMQDT3CAtB5Z">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-mQShdMiwodTYwvVHoqzQ3R" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">cat</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-mQShdMiwodTYwvVHoqzQ3R">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-aWC3HWmork2dkn8DCLZyoc" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">deer</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-aWC3HWmork2dkn8DCLZyoc">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-A5Wv5J5H5fS4kfCVzKZXvJ" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">dog</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-A5Wv5J5H5fS4kfCVzKZXvJ">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-6RfQqFcPTAqAvYA5C2BgSp" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">frog</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-6RfQqFcPTAqAvYA5C2BgSp">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-3grgWYoRXj7s3PnUa3eS7V" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">horse</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-3grgWYoRXj7s3PnUa3eS7V">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-D3SJiGkTVu4xeYC83sS3aK" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">ship</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-D3SJiGkTVu4xeYC83sS3aK">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-Qp39FURmf7sSoTwVPhgV63" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">truck</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-Qp39FURmf7sSoTwVPhgV63">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>
</div>
</div>





<div id="ipyplot-imgs-container-div-HsrYJwLSqbdaJxxbQi8oGe">
<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-j6yQagdiWEbv4JbtWDVnG9" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">airplane</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-j6yQagdiWEbv4JbtWDVnG9">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-exQ9mRosZ3hL3QjKo97cdd" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">automobile</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-exQ9mRosZ3hL3QjKo97cdd">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-fk8DbsRfpheMQDT3CAtB5Z" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">bird</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-fk8DbsRfpheMQDT3CAtB5Z">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-mQShdMiwodTYwvVHoqzQ3R" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">cat</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-mQShdMiwodTYwvVHoqzQ3R">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-aWC3HWmork2dkn8DCLZyoc" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">deer</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-aWC3HWmork2dkn8DCLZyoc">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-A5Wv5J5H5fS4kfCVzKZXvJ" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">dog</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-A5Wv5J5H5fS4kfCVzKZXvJ">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-6RfQqFcPTAqAvYA5C2BgSp" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">frog</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-6RfQqFcPTAqAvYA5C2BgSp">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-3grgWYoRXj7s3PnUa3eS7V" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">horse</h4>
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAJfUlEQVR4nAXB2Y+d510A4Hd/v/Xsc+ac8WzeMhk7cZOmWWqspJWolRYVpCKKEDdcIO75F/gfetVLBIirXACFiwKqVNoidYkoqG1ix/Z4Ns+c7dvf/cfz4N3bRwQ4Teje0Rxj9PzpeQgs7+d5P8oEnc9nm7pabtaj8cSsu/rVcpjns4MbtVPFcllXDUXMal+URTyMrbfWWh88BC84i6PIGMPAgg++8/byYj2dpBEjBMc8UL1uh1vJ7vY4jVlbrpCuj49vzB6+nsVSZlIHo/Vuuak4Ztfn189eBDHq0Yh6bOJeFEmRRylnLARgUjDw2HtAjk6HE7Vqu9pFNE6S5Pjozt3XDou64hFBBO69eXjzcMfoBogjFDHOg/G2MaaZfaCOMY9IQr2wJEGEY4E5wRgAWDpgLJDcR7GMsEEJi5Qq23oBCbk6j37lW2X0eDqd787mO5N4EAmEpECRoODBNhrFQgsCOhDPkMTxtO9i0NgAhhBCgMAO729LFVwFZ2eb3/3PkgDTZYtdR3R49vPiRDAHYbI9Xe/O0vBg2juezWeJBInBVF1tnClN/fy6vFqbSnXITl7bI8M4mmZ4QDHBnFD8vU/+pnl+9dN//dnFSXVVeu9JjKAfQ8r9OMoGeR8xiiilgtIMD3eHX3v88I3XD3uc2qJtFuXy5PLFb79YX14r3Z2XGxhkbNxP9of3vvEmT2Twgb3x1o0nnS7W7TjJnbWLajUfiDuDnCHPMRv2IhGnHpEoitMUF1er3/3zfw4uH0yHPadMMJh3IAO0mwUKyBftZlEl143dVPrtW/SQeYtYv88XiyUnaUajdegQKAF4P09jSQ1B2nRV0Yo4B44THE0nE8GgfXl5cXXtvCEkRkCZxPko1qVOZLSqi/bVqp/HGZaeOAOIxUJi56v1htCIYQuOOJdZy9MkcEqqqhFRnGcRF7RpauTZaJAqrb1HVreqWVVVm6RimGVXpYmiBEKljH15cnnz5fX0cNcHTZB13COOyKCfb036hDLtWaV80zbOGGfN9tZ4Phsh0NYa55w1JoSgus40bVtW5XLdFCVDxFpXN12rfWvhalE9++xlcIFxysrlulmuh0keCWm0Dcy3uFtrkvc4x7iXxoN+kmei2PhlWVCUbY1yhJBSGhkwJtS1qptaSuEJXlTVWmllg7L6/GxhtA0MWDDWVu0oy4tNed0Vk4PhMOWXp5c9NZeMj0eDLIkYDb1edH6imgaHEOq6VW0bDFqXalOZAIZdLkSe1sEVzmnAOmAVqAvgrWEMEY6Z6XRZ1R3YR994eP/e/Md/94PFWTfv9/p5ZozSzgVvtTbIh+VqhYKG4Js6bArlsSSMXy7L+aCHkrgKlQ7EYUqTzGOEMTAJyWzr9i/8qzVqd+5PH37t3uvHO+OE/ds//Hu5qdsmXS1KYzUwUmlcGzvstETeO7epWuOAi0hZu1aBG+ho1qHGoNC6muYySSMPwNrSEtnTMdo52Pv4Tz+4czQRMdx/dM8x9OPv/9OnT7/AmnkXkKCrTo+GEYtFV1ZVUTcGUcq0M4VSLaG/Obs+WZjKhwCgEe5N+lmarOqGnS4vf/Lrn2zd7n/3r75z694Es07rxhj/xjvHL3759If/+B/CpFb7AK4f4b35DYShNnqtwkZLghDnUHHNB8nL0+VlpSf70/PTa2cpwaJcV8ppMru96zLz1kdHd740A6aMV8ZbREFkbP/Nu7XAJYZXjS4M3Dy8eXjzFk+HDSSXLXlRhZPWPm82ekje+4OHvZ1tiNkf/fnv33/30IRwdnptNeBAyWA++su//ovfe/yOJRWijlAWx3kUxS7onYPZa8d3hZTgCcWxYdGnT1/892+/+PXp4um6OTfNmVmvWHf/62+/9833J/tbrfXpQHz7Ox+OJtmnP//f1XWZJT3S6CodRTTGAB4T4hy2jjhPWqUG2/m3//ibw61BmmeYiyUxbprUGdIDynaSeJfuPdj++M8ef/DxIzwgOzdHIfAnT57dPZoeHc2fPz09fX4hWcycM4EgBJ5Z6gAAMQBmnQISHNd7Dw7jWa/4zRlmfO/9m3/43ccXry6urjZVYx12N+aT/f2pYXbdLXcPRoykX3x2lv5J+MqX7/zql593jfE2MIyws5YxGgJqWw3AEAreWR5xQ1A8oNnO4LKp+v3e9Pawf5hFOwd38IHtTK108I4QjyFIKidb47wXCZ4mef9L790dfvKjYFEsGesMUEoEYw5Bq02nKkIIQpDSzGNCiBrMh45ywuVoNLTeGWSJ0xhZRLyxBgMGBIKKrDceTvj8xo4n6Xgf9m+PwWOGMVMWkRAsMtZqjEFI4Z0PAZQ2ygTLUN7PqKA8iiWf6DY4ooNuWaDBI0DYWdd2rSZitWo60yZpvFgVzvo07zeNb1vLGuOcNYyTqtrkabQ1HgMHAOiU6drO0+CDIwJv6vLFs/VwntO4Bm+DpZXqlNEAYK11HE5eXhRVSTgp65qA6BR8/uSsKC2r6lpwIRkXQhLMMGbGqLZtrfUIECBkwdGIbDbrf/nBD3vjbx3eyjyyzru201VdO+e44CTwi1dL4zyTzDjvjXYhnJ+cL5c1i6WIIiE4iYZ9yUTXqWJTdF2bZT0Ivm1bRFDaT95+98vPX37+/e/97Ucfvvf6g73+tgSgjEYYeWfcdbF58vQ5IsiD8wF3xsQZ4RVrOkM48sSbiCJBAEIIHqSM+v1BluWcCYxxlEQOudtHB4+/9ahYdp/8/X/94qeflVXZKW2tByAA+OpqWdVq72C/qqvLq+umrftjtrs/rZuGOaOcAUZRksScC0qY4AIAtNLBeOK5095avVovv/rh8fuPvvKzH/3fsxens5dSZlm/PzLWlGVT1e3de7cHg1lvSDdFSQndv3tDtaQ1DWtaa521jhiDkxi89wgwpcwbbzvb1u7V2XJ7azLsD1qrD97cWqstwUhdIkuciJ13wGSyfWP38JY0xmGCjKVFWaRZHEfAEs42RYcQ8t60ncfBadVRymQUCSHrVlkH+Sj/6kfv7B/OCXf5KH3r3XuJSHq9nkYdJQwzIglFgJRR1toojvM8F1JSwYzWQkoWkOCMI8LrpvNGN3VDGRkOKGURkjJK+EywdFLHOfGBsCDZkKcy5YzZThOPnfVlVWijMSNMMAhIRpJx3rSaEFlXihkLztqus03TSi4oSylDgKl2XvtgTQPIyx5zWBnlvQ660YYazvhidTUaDgLA4uJaGTOZzzzGq3KNEBDKLs7XIYAP/v8BDjUbSftL/RoAAAAASUVORK5CYII="/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-3grgWYoRXj7s3PnUa3eS7V">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-D3SJiGkTVu4xeYC83sS3aK" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">ship</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-D3SJiGkTVu4xeYC83sS3aK">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>

<div class="ipyplot-placeholder-div-HsrYJwLSqbdaJxxbQi8oGe">
    <div id="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-Qp39FURmf7sSoTwVPhgV63" class="ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe">
        <h4 style="font-size: 12px; word-wrap: break-word;">truck</h4>
        <img src="data:image/png;base64,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"/>
        <a href="#!">
            <span class="ipyplot-img-close"/>
        </a>
        <a href="#ipyplot-content-div-HsrYJwLSqbdaJxxbQi8oGe-Qp39FURmf7sSoTwVPhgV63">
            <span class="ipyplot-img-expand"/>
        </a>
    </div>
</div>
</div>



<br>
<br>

# 2. Image Recognition Architectures
 
 
## 2.1 GoogLeNet

The LeNet-5 convolutional neural network (introduced in 1998 by Yann LeCun et al. in the paper “Gradient-Based Learning Applied To Document Recognition”) was the  first paper which showed utilisation of convolutional neural networks for the computer vision task of image classification. Following that in 2012 - AlexNet came (I covered in my previous report), a convolutional neural network architecture introduced the composition of consecutively stacked convolutional layers. The creators of AlexNet trained the network using graphical processing units (GPUs).

Following these 2 papers, efficient computing resources and intuitive CNN architectures led to the rapid development of competetive solutions to lot of computer vision tasks. Although, researchers discovered that an increase of layers and units within a network led to a significant performance gain. But they also foudn out that - increasing the layers to create more extensive networks came at a cost. Large networks are prone to overfitting and suffer from either exploding or vanishing gradient problem.

The GoogLeNet architecture (introduced in 2015 - _"Going Deeper with Convolutions"_) solved most of the problems that large networks faced, mainly through the Inception module's utilisation.


>![m](https://qph.cf2.quoracdn.net/main-qimg-a84c42b1e8664cb4156e075141f3f851)
<br>
> Figure. Inception module 

At the time researchers were a bit confused on exactly what kernel sizes to use for their convolutional networks. The Inception module is a neural network architecture that leverages feature detection at different scales through convolutions with different kernel sizes and reduced the computational cost of training an extensive network through dimensional reduction.

Below I implement the _inception_ module with a slight modiifcation of adding batch-normalization to stabilize the training (the original implementation doesn't contain batch-normalization):


```python
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)
```

> ![1](https://miro.medium.com/max/720/1*66hY3zZTf0Lw2ItybiRxyg.png)
> Figure. GoogLeNet architecture

The GoogLeNet architecture consists of 22 layers (27 layers including pooling layers), and part of these layers are a total of 9 inception modules. Following is the implementation of whole GoogLeNet architecture:


```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
g_net = GoogLeNet()
g_net = g_net.to(device)
```

### Hyper-parameters:

- I set the optimization criterion to Cross Entropy - a standard classification loss.
- I initialize stochastic gradient descent (SGD) optimizer with a learning rate of 1e-1.
- Additionally, I apply a learning rate scheduler which reduces the learning rate using [cosine annealing](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjh5oXDt7r6AhUlL30KHb1sBz0QFnoECAkQAQ&url=https%3A%2F%2Fpaperswithcode.com%2Fmethod%2Fcosine-annealing&usg=AOvVaw1c5WijiHqMcnRm6311R1w_) if the performance stagnates.


```python
if device == 'cuda':
    net = torch.nn.DataParallel(g_net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(g_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
```

## Testing

In the following function, I'll be switching off the gradient computation using `torch.no_grad` and doing the evaluation on test set. I do the following:
- Compute the loss and accuracy over the test set.
- Print the logs of evaluation if promted (controlled with log_epochs).
- Save the best performing model checkpoint


```python
def test(net, epoch, total_epochs, log_epochs=None, save_ckpt=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)  # test loss
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)   # total predictions
            correct += predicted.eq(targets).sum().item()  # correct predictions
        
        # Test-set evaluation logs     
        if epoch%log_epochs==0:
            print('\t     Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/save_ckpt.pth')
        best_acc = acc
    
    return test_loss, (100.*correct/total)
```

## Training
The below function is used to train the GoogLeNet. After each forward-backward propagation, it accumulates the per-batch losses and accuracies and prints the logs if prompted (controlled with `log_epochs`). 


```python
def train(net, epoch, total_epochs, log_epochs=None):
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_loss/=(batch_idx+1)
    if epoch%log_epochs==0:
            print(f'Epoch: {epoch} -> Train Loss: {train_loss:.3f} | Train Acc: {100.*correct/total} ({correct}/{total})')

    return train_loss, (100.*correct/total)
```

Now that we have all the components for training our implementation of GoogLeNet ready, we can start the training procedure. For each iteration, I'm training GoogLeNet on the training samples and computing the loss and accuracy values. It is followed by evaluation on the test-set. All the loss and accuracy values are stored and dumped for later use.


```python
%%capture gnet_stdout 

gnet_logs = {'tl':[], 'ta':[], 'vl':[], 'va':[]}
total_epochs = 200

for epoch in range(start_epoch, start_epoch+total_epochs):
    
    train_loss, train_acc = train(g_net, epoch, total_epochs, log_epochs=10)
    test_loss, test_acc = test(g_net, epoch, total_epochs, log_epochs=10)
    
    train_loss/=len(trainloader)
    test_loss/=len(testloader)
    gnet_logs['tl'].append(train_loss)
    gnet_logs['ta'].append(train_acc)
    gnet_logs['vl'].append(test_loss)
    gnet_logs['va'].append(test_acc)
    scheduler.step()

# save logs to file
with open('Gnet_logs.pkl', 'wb') as f:
    pickle.dump(gnet_logs, f)
        
```


```python
gnet_stdout()
```

    Epoch: 0 -> Train Loss: 1.580 | Train Acc: 41.714 (20857/50000)
    	     Test Loss: 1.282 | Test Acc: 54.470% (5447/10000)
    Epoch: 10 -> Train Loss: 0.420 | Train Acc: 85.678 (42839/50000)
    	     Test Loss: 0.621 | Test Acc: 79.500% (7950/10000)
    Epoch: 20 -> Train Loss: 0.341 | Train Acc: 88.306 (44153/50000)
    	     Test Loss: 0.554 | Test Acc: 81.830% (8183/10000)
    Epoch: 30 -> Train Loss: 0.305 | Train Acc: 89.578 (44789/50000)
    	     Test Loss: 0.650 | Test Acc: 79.600% (7960/10000)
    Epoch: 40 -> Train Loss: 0.292 | Train Acc: 89.894 (44947/50000)
    	     Test Loss: 0.488 | Test Acc: 84.040% (8404/10000)
    Epoch: 50 -> Train Loss: 0.264 | Train Acc: 90.854 (45427/50000)
    	     Test Loss: 0.763 | Test Acc: 77.520% (7752/10000)
    Epoch: 60 -> Train Loss: 0.243 | Train Acc: 91.684 (45842/50000)
    	     Test Loss: 0.595 | Test Acc: 81.030% (8103/10000)
    Epoch: 70 -> Train Loss: 0.230 | Train Acc: 92.022 (46011/50000)
    	     Test Loss: 0.697 | Test Acc: 79.470% (7947/10000)
    Epoch: 80 -> Train Loss: 0.205 | Train Acc: 92.918 (46459/50000)
    	     Test Loss: 0.639 | Test Acc: 81.260% (8126/10000)
    Epoch: 90 -> Train Loss: 0.182 | Train Acc: 93.718 (46859/50000)
    	     Test Loss: 0.673 | Test Acc: 80.030% (8003/10000)
    Epoch: 100 -> Train Loss: 0.154 | Train Acc: 94.742 (47371/50000)
    	     Test Loss: 0.476 | Test Acc: 85.760% (8576/10000)
    Epoch: 110 -> Train Loss: 0.124 | Train Acc: 95.772 (47886/50000)
    	     Test Loss: 0.420 | Test Acc: 87.380% (8738/10000)
    Epoch: 120 -> Train Loss: 0.102 | Train Acc: 96.636 (48318/50000)
    	     Test Loss: 0.356 | Test Acc: 89.270% (8927/10000)
    Epoch: 130 -> Train Loss: 0.071 | Train Acc: 97.654 (48827/50000)
    	     Test Loss: 0.652 | Test Acc: 82.360% (8236/10000)
    Epoch: 140 -> Train Loss: 0.052 | Train Acc: 98.326 (49163/50000)
    	     Test Loss: 0.295 | Test Acc: 91.370% (9137/10000)
    Epoch: 150 -> Train Loss: 0.021 | Train Acc: 99.444 (49722/50000)
    	     Test Loss: 0.249 | Test Acc: 93.090% (9309/10000)
    Epoch: 160 -> Train Loss: 0.003 | Train Acc: 99.994 (49997/50000)
    	     Test Loss: 0.159 | Test Acc: 95.180% (9518/10000)
    Epoch: 170 -> Train Loss: 0.003 | Train Acc: 100.0 (50000/50000)
    	     Test Loss: 0.152 | Test Acc: 95.150% (9515/10000)
    Epoch: 180 -> Train Loss: 0.003 | Train Acc: 99.998 (49999/50000)
    	     Test Loss: 0.153 | Test Acc: 95.120% (9512/10000)
    Epoch: 190 -> Train Loss: 0.002 | Train Acc: 100.0 (50000/50000)
    	     Test Loss: 0.149 | Test Acc: 95.340% (9534/10000)
    

# 2.2 ResNet-18

Following GoogLeNet, lot of researchers starting working with deeper and more complex models. One of the complication they faced with going deep with convlutional nets was that - with the network depth increasing, accuracy gets saturated and then degrades rapidly.


In an attempt to solve this, a set of researchers came up with the following idea: Instead of learning a direct mapping of $$x\rightarrow y$$ with a function $$H(x)$$ (A few stacked non-linear layers). Let us define the residual function using $$F(x) = H(x) — x$$, which can be reframed into $$H(x) = F(x)+x$$, where $$F(x)$$ and $$x$$ represents the stacked non-linear layers and the identity function(input=output) respectively. 
> ![1_RTYKpn1Vqr-8zT5fqa8-jA.png](https://miro.medium.com/max/640/1*RTYKpn1Vqr-8zT5fqa8-jA.png)

The ResNet-18 architecture uses such residual connections such that the input passed via the shortcut matches is resized to dimensions of the main path's output. Following shows the overall structure of ResNet-18 architecture:

> ![bpo9v4r.png](https://i.imgur.com/bpo9v4r.png)

Here I implement the ResNet-18 architecture. The `Basicblock` defines the block before and after which there'll be residual connections (shortcut). The `ResNet` classs compiles several such `Basicblock(s)` and adds residuals to them (resizes the residual inputs if required).  


```python
# Basic block before and after which there are shortcut connections. 
# Batch-norm is a modification I added for stability.
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Resnet class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    # resizing the shortcuts with strides whenever required before adding them
    # to the main path 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Initiating the resnet class with blocks as defined in the original paper.
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

res_net = ResNet18()
res_net = res_net.to(device)
```

### Hyperparameters
Same set of Hyperparameters are used as for fair-comparison


```python
if device == 'cuda':
    net = torch.nn.DataParallel(res_net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```


```python
%%capture resnet_stdout

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epochs = 101

resnet_logs = {'tl':[], 'ta':[], 'vl':[], 'va':[]}

for epoch in range(start_epoch, start_epoch+total_epochs):
    
    train_loss, train_acc = train(res_net, epoch, total_epochs, log_epochs=10)
    test_loss, test_acc = test(res_net, epoch, total_epochs, log_epochs=10, save_ckpt="resnet")
    
    train_loss/=len(trainloader)
    test_loss/=len(testloader)
    resnet_logs['tl'].append(train_loss)
    resnet_logs['ta'].append(train_acc)
    resnet_logs['vl'].append(test_loss)
    resnet_logs['va'].append(test_acc)
    scheduler.step()

with open('resnet_logs.pkl', 'wb') as f:
    pickle.dump(resnet_logs, f)
    
```


```python
resnet_stdout()
```

    Epoch: 0 -> Train Loss: 2.043 | Train Acc: 26.122 (13061/50000)
    	     Test Loss: 1.612 | Test Acc: 39.590% (3959/10000)
    Epoch: 10 -> Train Loss: 0.507 | Train Acc: 82.658 (41329/50000)
    	     Test Loss: 0.711 | Test Acc: 76.380% (7638/10000)
    Epoch: 20 -> Train Loss: 0.391 | Train Acc: 86.676 (43338/50000)
    	     Test Loss: 0.558 | Test Acc: 81.650% (8165/10000)
    Epoch: 30 -> Train Loss: 0.348 | Train Acc: 88.074 (44037/50000)
    	     Test Loss: 0.510 | Test Acc: 82.830% (8283/10000)
    Epoch: 40 -> Train Loss: 0.322 | Train Acc: 89.028 (44514/50000)
    	     Test Loss: 0.447 | Test Acc: 85.120% (8512/10000)
    Epoch: 50 -> Train Loss: 0.304 | Train Acc: 89.698 (44849/50000)
    	     Test Loss: 0.552 | Test Acc: 82.260% (8226/10000)
    Epoch: 60 -> Train Loss: 0.284 | Train Acc: 90.42 (45210/50000)
    	     Test Loss: 0.514 | Test Acc: 82.970% (8297/10000)
    Epoch: 70 -> Train Loss: 0.262 | Train Acc: 91.034 (45517/50000)
    	     Test Loss: 0.432 | Test Acc: 85.380% (8538/10000)
    Epoch: 80 -> Train Loss: 0.240 | Train Acc: 91.824 (45912/50000)
    	     Test Loss: 0.391 | Test Acc: 87.020% (8702/10000)
    Epoch: 90 -> Train Loss: 0.216 | Train Acc: 92.568 (46284/50000)
    	     Test Loss: 0.305 | Test Acc: 90.260% (9026/10000)
    Epoch: 100 -> Train Loss: 0.185 | Train Acc: 93.634 (46817/50000)
    	     Test Loss: 0.345 | Test Acc: 89.250% (8925/10000)
    

# 2.3 PreAct-R18 (Identity Mappings in Deep Residual Networks)

This is a follow-up work to the ResNet-18. As we discussed in the lecture, residual block can be represented with the equations $$y_l = h(x_l) + F(x_l, W_l)$$; $$x_{l+1} = f(y_l)$$, where $$x_l$$ is the input to the $$l-th$$ unit and $$x_{l+1}$$ is the output of the $$l-th$$ unit. In the original ResNet-18, $$h(x_l) = x_l$$, $$f$$ is ReLu, and $$F$$ consists of 2-3 convolutional layers (basic-block based architecture) with BN and ReLU in between. In this, they propose a residual block with both $$h(x)$$ and $$f(x)$$ as identity mappings involving BN and ReLU before the actual convolutions for efficient training and better performance.


![a](/assets/blog/recognition/preact_resnet.png)

```python
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock. The modification is addition of BatchNorm and ReLU before convolutions'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

# Quite similar to the ResNet-18 class except it utilizes the modified PreActBlock
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Modified PreAct-ResNet-18
def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])


preres_net = PreActResNet18()
preres_net = preres_net.to(device)
```

### Hyper-parameters
Same set of Hyperparameters are used as for fair-comparison


```python
if device == 'cuda':
    net = torch.nn.DataParallel(preres_net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(preres_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```


```python
%%capture preresnet_stdout

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epochs = 101

preresnet_logs = {'tl':[], 'ta':[], 'vl':[], 'va':[]}

for epoch in range(start_epoch, start_epoch+total_epochs):
    
    train_loss, train_acc = train(preres_net, epoch, total_epochs, log_epochs=10)
    test_loss, test_acc = test(preres_net, epoch, total_epochs, log_epochs=10, save_ckpt="preresnet")
    
    train_loss/=len(trainloader)
    test_loss/=len(testloader)
    preresnet_logs['tl'].append(train_loss)
    preresnet_logs['ta'].append(train_acc)
    preresnet_logs['vl'].append(test_loss)
    preresnet_logs['va'].append(test_acc)
    scheduler.step()

with open('preresnet_logs.pkl', 'wb') as f:
    pickle.dump(preresnet_logs, f)
```


```python
preresnet_stdout()
```

    Epoch: 0 -> Train Loss: 4.213 | Train Acc: 10.334 (5167/50000)
    	     Test Loss: 3.097 | Test Acc: 8.150% (815/10000)
    Epoch: 10 -> Train Loss: 1.022 | Train Acc: 63.372 (31686/50000)
    	     Test Loss: 1.041 | Test Acc: 64.310% (6431/10000)
    Epoch: 20 -> Train Loss: 0.575 | Train Acc: 80.43 (40215/50000)
    	     Test Loss: 0.721 | Test Acc: 75.040% (7504/10000)
    Epoch: 30 -> Train Loss: 0.483 | Train Acc: 83.444 (41722/50000)
    	     Test Loss: 1.626 | Test Acc: 61.830% (6183/10000)
    Epoch: 40 -> Train Loss: 0.433 | Train Acc: 85.166 (42583/50000)
    	     Test Loss: 0.535 | Test Acc: 82.670% (8267/10000)
    Epoch: 50 -> Train Loss: 0.389 | Train Acc: 86.586 (43293/50000)
    	     Test Loss: 0.490 | Test Acc: 83.660% (8366/10000)
    Epoch: 60 -> Train Loss: 0.361 | Train Acc: 87.634 (43817/50000)
    	     Test Loss: 0.601 | Test Acc: 81.710% (8171/10000)
    Epoch: 70 -> Train Loss: 0.332 | Train Acc: 88.66 (44330/50000)
    	     Test Loss: 0.459 | Test Acc: 84.960% (8496/10000)
    Epoch: 80 -> Train Loss: 0.307 | Train Acc: 89.432 (44716/50000)
    	     Test Loss: 0.492 | Test Acc: 84.460% (8446/10000)
    Epoch: 90 -> Train Loss: 0.277 | Train Acc: 90.504 (45252/50000)
    	     Test Loss: 0.376 | Test Acc: 86.950% (8695/10000)
    Epoch: 100 -> Train Loss: 0.241 | Train Acc: 91.75 (45875/50000)
    	     Test Loss: 0.428 | Test Acc: 86.060% (8606/10000)
    

# 3. Inference and Evaluation

## 3.1 Training Analysis

### Loss comparison


```python
plt.figure(figsize=(18, 8))

plt.plot(gnet_logs['vl'][:101], color='blue', label='GoogLeNet')

plt.plot(resnet_logs['vl'][:101], color='green', label='ResNet-18')

plt.plot(preresnet_logs['vl'][:101], color='red', label='PreAct-R18')


plt.xlabel("# of Iterations", fontsize=14)
plt.ylabel("Loss Value", fontsize=14)
plt.grid(ls='--', c='grey', alpha=0.5)
plt.title("Comparison of Test-set Losses", fontsize=14)
plt.legend(fontsize=16)
plt.show()

```


    
![png](/assets/blog/recognition/output_47_0.png)
    


### Accuracy comparison


```python
plt.figure(figsize=(18, 8))

plt.plot(gnet_logs['va'][:101], color='blue', label='GoogLeNet')

plt.plot(resnet_logs['va'][:101], color='green', label='ResNet-18')

plt.plot(preresnet_logs['va'][:101], color='red', label='PreAct-R18')


plt.xlabel("# of Iterations", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(ls='--', c='grey', alpha=0.5)
plt.title("Comparison of Test-set Accuracies", fontsize=14)
plt.legend(fontsize=16)
plt.show()

```


    
![png](/assets/blog/recognition/output_49_0.png)
    


Both the test-set losses and accuracies seems to be converging for all the models. Even though the actual values themselves seems a changing a lot, however, it could be agreed upon that overall the models are still getting improved the best loss and accuracy values are still improving. 

All 3 models are observed to perform very similar to each other. On one hand, GoogLeNet falls a little short in terms as it performs worse than both RestNet-18 and PreAct-R18. Among the rest two, its a close match - ResNet-18 dominates PreAct-R18 overs some iterations whereas the opposite is observed in some other iterations.   

- **`get_predictions -`** Function to generate predictions from a data iterator. Since the model outputs class probabilities, the class predictions can be obtained by considering the index with highest probability.  


```python
def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


```

### Generating predictions



```python
images, g_net_labels, g_net_probs = get_predictions(g_net, testloader, device)
g_net_pred_labels = torch.argmax(g_net_probs, 1)

images, resnet_labels, resnet_probs = get_predictions(res_net, testloader, device)
resnet_pred_labels = torch.argmax(resnet_probs, 1)

images, preresnet_labels, preresnet_probs = get_predictions(preres_net, testloader, device)
preresnet_pred_labels = torch.argmax(preresnet_probs, 1)

```

### Confusion Matrix

One ways of assessing model performance is to evaluate the statistics of correctness of its predictions. Confusion matrix allows us to visualize the confidence of model predictions. 


```python
def plot_confusion_matrix(labels, pred_labels, classes, title=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='magma_r', ax=ax)
    plt.xticks(rotation=20)
    plt.title(title)
    plt.show()
    
plot_confusion_matrix(g_net_labels, g_net_pred_labels, classes, title="GoogLeNet performance")
plot_confusion_matrix(resnet_labels, resnet_pred_labels, classes, title="ResNet-18 performance")
plot_confusion_matrix(preresnet_labels, preresnet_pred_labels, classes, title="PreAct-R18 performance")

```


    
![png](/assets/blog/recognition/output_56_0.png)
    



    
![png](/assets/blog/recognition/output_56_1.png)
    



    
![png](/assets/blog/recognition/output_56_2.png)
    


From the confusion matrix, models seem to get mixed up the most between cats and dogs. Another such case is automobile and trucks which is agreeable as trucks and automobiles have bery similar features. One more example is planes and birds, which also from semantics perspective can become hard to recognize due to similar visual features and action of flying.

## Discussion

One interesting thing of observation from confusion matrices was that, GoogLeNet (trained for 200 epochs) outperforms both ResNet-18 and PreAct-R18 (both trained for 101 epochs). This wasn't the case when we performed the training analysis (100 epochs) since the later models are more advanced. What this suggests is that both ResNet-18 and PreAact-R18 have more potential to get trained further and outperform the GoogLeNet, however due to limited time constraints I couldn't train them further. 

The modification to ResNet-18 in the model PreAct-R18, certainly brings some performance gains (observe accuracies for cats, ship etc) where the ResNet-18 model seemed to be least performing. While the modification improves where ResNet-18 performs the poorest (cats), it lacks in some other classes (like dogs) where a drop in performance is observed. Going through the original paper of PreAct-R18 (_Identity Mappings in Deep Residual Networks_), the authors utilize a depth of 1000 layers to demonstrate the performance gains. It could be that going further deeper, PreAct-R18 should widen the performance gap when trained for more iterations. However, for more shallower networks (~18-22 layers) my observations suggest all the above methods performs very similar and not a significant performance gains are observed  

# References

```
[1] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[3] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.

[4] Deep Learning: GoogLeNet Explained - Medium

[5] kuangliu: pytorch-cifar - 2021 Github 
```


```python

```
