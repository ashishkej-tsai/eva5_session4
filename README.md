# EVA5 Session4 Assignment Solution

## Description
Aim of this assignment is to achieve 99.4% test accuracy on MNIST dataset with Less than 20K Parameters, Less than 20 Epochs, No fully connected layer.

## Code Walk Through
We are using Pytorch library to build and train our network. Main libraries are torch and torchvision.  
torch.nn --> used for 2d Convolutional Layer (Conv2d), Batch Normalisation (BatchNorm2d)  
torch.nn.functional --> used for ReLU activation function (relu)  
torch.optim --> SGD optimiser used in Backpropagation  

torchvision library is used to Download MNIST dataset, data generator and transforms for Data Augmentation.  

The Architecture:

self.conv1 = nn.Conv2d(1, 8, 3)   #26x26
self.bn1   = nn.BatchNorm2d(8)
self.conv2 = nn.Conv2d(8, 16, 3)  #24X24
self.bn2   = nn.BatchNorm2d(16)
self.conv3 = nn.Conv2d(16, 16, 3) #22x22
self.bn3   = nn.BatchNorm2d(16)
self.pool1 = nn.MaxPool2d(2, 2)   #11x11
self.conv4 = nn.Conv2d(16, 16, 3)   #9x9
self.bn4   = nn.BatchNorm2d(16)
self.conv5 = nn.Conv2d(16, 32, 3)  #7x7
self.bn5   = nn.BatchNorm2d(32)
self.conv6 = nn.Conv2d(32, 16, 3) #5x5
self.bn6   = nn.BatchNorm2d(16)
self.conv7 = nn.Conv2d(16, 10, 5) #1x1
self.bn7   = nn.BatchNorm2d(10)
  
x = F.relu(self.bn1(self.conv1(x)))
x = F.relu(self.bn2(self.conv2(x)))
x = F.relu(self.bn3(self.conv3(x)))
x = self.pool1(x)
x = F.relu(self.bn4(self.conv4(x)))
x = F.relu(self.bn5(self.conv5(x)))
x = F.relu(self.bn6(self.conv6(x)))
x = self.conv7(x)
x = x.view(-1, 10)
return F.log_softmax(x)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
            Conv2d-3           [-1, 16, 24, 24]           1,168
       BatchNorm2d-4           [-1, 16, 24, 24]              32
            Conv2d-5           [-1, 16, 22, 22]           2,320
       BatchNorm2d-6           [-1, 16, 22, 22]              32
         MaxPool2d-7           [-1, 16, 11, 11]               0
            Conv2d-8             [-1, 16, 9, 9]           2,320
       BatchNorm2d-9             [-1, 16, 9, 9]              32
           Conv2d-10             [-1, 32, 7, 7]           4,640
      BatchNorm2d-11             [-1, 32, 7, 7]              64
           Conv2d-12             [-1, 16, 5, 5]           4,624
      BatchNorm2d-13             [-1, 16, 5, 5]              32
           Conv2d-14             [-1, 10, 1, 1]           4,010
================================================================
Total params: 19,370
Trainable params: 19,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.41
Params size (MB): 0.07
Estimated Total Size (MB): 0.48

## Test Accuracy
Reached 99.49% test accuracy in 15th epoch

## Conclusion
Batch Normalisation after each layer and Data Augmentation (RandomRotation, RandomAffine, ColorJitter) helped to achieve the test accuracy. To reduce the parameters FC layers have not been used.
