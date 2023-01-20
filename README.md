# Facial Landmark Detection

### Dataset
- [FFHQ 512x512](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)
- the landmarks labels were made by using dlib library (utils.py)      

### Model ([Xception](https://arxiv.org/abs/1610.02357))
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 256, 256]             864
       BatchNorm2d-2         [-1, 32, 256, 256]              64
              ReLU-3         [-1, 32, 256, 256]               0
            Conv2d-4         [-1, 64, 256, 256]          18,432
       BatchNorm2d-5         [-1, 64, 256, 256]             128
              ReLU-6         [-1, 64, 256, 256]               0
            Conv2d-7        [-1, 128, 128, 128]           8,192
       BatchNorm2d-8        [-1, 128, 128, 128]             256
            Conv2d-9        [-1, 128, 256, 256]           8,192
           Conv2d-10        [-1, 128, 256, 256]           1,152
DepthwiseSeparableConv2d-11        [-1, 128, 256, 256]               0
      BatchNorm2d-12        [-1, 128, 256, 256]             256
             ReLU-13        [-1, 128, 256, 256]               0
           Conv2d-14        [-1, 128, 256, 256]          16,384
           Conv2d-15        [-1, 128, 256, 256]           1,152
DepthwiseSeparableConv2d-16        [-1, 128, 256, 256]               0
      BatchNorm2d-17        [-1, 128, 256, 256]             256
        MaxPool2d-18        [-1, 128, 128, 128]               0
    ResidualBlock-19        [-1, 128, 128, 128]               0
           Conv2d-20          [-1, 256, 64, 64]          32,768
      BatchNorm2d-21          [-1, 256, 64, 64]             512
           Conv2d-22        [-1, 256, 128, 128]          32,768
           Conv2d-23        [-1, 256, 128, 128]           2,304
DepthwiseSeparableConv2d-24        [-1, 256, 128, 128]               0
      BatchNorm2d-25        [-1, 256, 128, 128]             512
             ReLU-26        [-1, 256, 128, 128]               0
           Conv2d-27        [-1, 256, 128, 128]          65,536
           Conv2d-28        [-1, 256, 128, 128]           2,304
DepthwiseSeparableConv2d-29        [-1, 256, 128, 128]               0
      BatchNorm2d-30        [-1, 256, 128, 128]             512
        MaxPool2d-31          [-1, 256, 64, 64]               0
    ResidualBlock-32          [-1, 256, 64, 64]               0
           Conv2d-33          [-1, 512, 32, 32]         131,072
      BatchNorm2d-34          [-1, 512, 32, 32]           1,024
           Conv2d-35          [-1, 512, 64, 64]         131,072
           Conv2d-36          [-1, 512, 64, 64]           4,608
DepthwiseSeparableConv2d-37          [-1, 512, 64, 64]               0
      BatchNorm2d-38          [-1, 512, 64, 64]           1,024
             ReLU-39          [-1, 512, 64, 64]               0
           Conv2d-40          [-1, 512, 64, 64]         262,144
           Conv2d-41          [-1, 512, 64, 64]           4,608
DepthwiseSeparableConv2d-42          [-1, 512, 64, 64]               0
      BatchNorm2d-43          [-1, 512, 64, 64]           1,024
        MaxPool2d-44          [-1, 512, 32, 32]               0
    ResidualBlock-45          [-1, 512, 32, 32]               0
        EntryFlow-46          [-1, 512, 32, 32]               0
             ReLU-47          [-1, 512, 32, 32]               0
           Conv2d-48          [-1, 512, 32, 32]         262,144
           Conv2d-49          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-50          [-1, 512, 32, 32]               0
      BatchNorm2d-51          [-1, 512, 32, 32]           1,024
             ReLU-52          [-1, 512, 32, 32]               0
           Conv2d-53          [-1, 512, 32, 32]         262,144
           Conv2d-54          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-55          [-1, 512, 32, 32]               0
      BatchNorm2d-56          [-1, 512, 32, 32]           1,024
             ReLU-57          [-1, 512, 32, 32]               0
           Conv2d-58          [-1, 512, 32, 32]         262,144
           Conv2d-59          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-60          [-1, 512, 32, 32]               0
      BatchNorm2d-61          [-1, 512, 32, 32]           1,024
    ResidualBlock-62          [-1, 512, 32, 32]               0
             ReLU-63          [-1, 512, 32, 32]               0
           Conv2d-64          [-1, 512, 32, 32]         262,144
           Conv2d-65          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-66          [-1, 512, 32, 32]               0
      BatchNorm2d-67          [-1, 512, 32, 32]           1,024
             ReLU-68          [-1, 512, 32, 32]               0
           Conv2d-69          [-1, 512, 32, 32]         262,144
           Conv2d-70          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-71          [-1, 512, 32, 32]               0
      BatchNorm2d-72          [-1, 512, 32, 32]           1,024
             ReLU-73          [-1, 512, 32, 32]               0
           Conv2d-74          [-1, 512, 32, 32]         262,144
           Conv2d-75          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-76          [-1, 512, 32, 32]               0
      BatchNorm2d-77          [-1, 512, 32, 32]           1,024
    ResidualBlock-78          [-1, 512, 32, 32]               0
             ReLU-79          [-1, 512, 32, 32]               0
           Conv2d-80          [-1, 512, 32, 32]         262,144
           Conv2d-81          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-82          [-1, 512, 32, 32]               0
      BatchNorm2d-83          [-1, 512, 32, 32]           1,024
             ReLU-84          [-1, 512, 32, 32]               0
           Conv2d-85          [-1, 512, 32, 32]         262,144
           Conv2d-86          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-87          [-1, 512, 32, 32]               0
      BatchNorm2d-88          [-1, 512, 32, 32]           1,024
             ReLU-89          [-1, 512, 32, 32]               0
           Conv2d-90          [-1, 512, 32, 32]         262,144
           Conv2d-91          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-92          [-1, 512, 32, 32]               0
      BatchNorm2d-93          [-1, 512, 32, 32]           1,024
    ResidualBlock-94          [-1, 512, 32, 32]               0
             ReLU-95          [-1, 512, 32, 32]               0
           Conv2d-96          [-1, 512, 32, 32]         262,144
           Conv2d-97          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-98          [-1, 512, 32, 32]               0
      BatchNorm2d-99          [-1, 512, 32, 32]           1,024
            ReLU-100          [-1, 512, 32, 32]               0
          Conv2d-101          [-1, 512, 32, 32]         262,144
          Conv2d-102          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-103          [-1, 512, 32, 32]               0
     BatchNorm2d-104          [-1, 512, 32, 32]           1,024
            ReLU-105          [-1, 512, 32, 32]               0
          Conv2d-106          [-1, 512, 32, 32]         262,144
          Conv2d-107          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-108          [-1, 512, 32, 32]               0
     BatchNorm2d-109          [-1, 512, 32, 32]           1,024
   ResidualBlock-110          [-1, 512, 32, 32]               0
      MiddleFlow-111          [-1, 512, 32, 32]               0
          Conv2d-112         [-1, 1024, 16, 16]         524,288
     BatchNorm2d-113         [-1, 1024, 16, 16]           2,048
            ReLU-114          [-1, 512, 32, 32]               0
          Conv2d-115          [-1, 512, 32, 32]         262,144
          Conv2d-116          [-1, 512, 32, 32]           4,608
DepthwiseSeparableConv2d-117          [-1, 512, 32, 32]               0
     BatchNorm2d-118          [-1, 512, 32, 32]           1,024
            ReLU-119          [-1, 512, 32, 32]               0
          Conv2d-120         [-1, 1024, 32, 32]         524,288
          Conv2d-121         [-1, 1024, 32, 32]           9,216
DepthwiseSeparableConv2d-122         [-1, 1024, 32, 32]               0
     BatchNorm2d-123         [-1, 1024, 32, 32]           2,048
       MaxPool2d-124         [-1, 1024, 16, 16]               0
   ResidualBlock-125         [-1, 1024, 16, 16]               0
          Conv2d-126         [-1, 1024, 16, 16]       1,048,576
          Conv2d-127         [-1, 1024, 16, 16]           9,216
DepthwiseSeparableConv2d-128         [-1, 1024, 16, 16]               0
     BatchNorm2d-129         [-1, 1024, 16, 16]           2,048
            ReLU-130         [-1, 1024, 16, 16]               0
          Conv2d-131         [-1, 1024, 16, 16]       1,048,576
          Conv2d-132         [-1, 1024, 16, 16]           9,216
DepthwiseSeparableConv2d-133         [-1, 1024, 16, 16]               0
     BatchNorm2d-134         [-1, 1024, 16, 16]           2,048
            ReLU-135         [-1, 1024, 16, 16]               0
AdaptiveAvgPool2d-136           [-1, 1024, 1, 1]               0
        ExitFlow-137           [-1, 1024, 1, 1]               0
         Flatten-138                 [-1, 1024]               0
          Linear-139                 [-1, 1024]       1,049,600
            ReLU-140                 [-1, 1024]               0
          Linear-141                  [-1, 136]         139,400
================================================================
Total params: 8,580,776
Trainable params: 8,580,776
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 1612.04
Params size (MB): 32.73
Estimated Total Size (MB): 1647.77
----------------------------------------------------------------
```

### Result
![facial_landmark_detection_epoch_49](https://user-images.githubusercontent.com/66504341/213594217-b61e963f-04cf-4be1-b9a1-bfad91d86bdd.png)
