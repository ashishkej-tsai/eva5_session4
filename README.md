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

self.conv1 = nn.Conv2d(1, 8, 3)      
self.bn1   = nn.BatchNorm2d(8)  
self.conv2 = nn.Conv2d(8, 16, 3)  
self.bn2   = nn.BatchNorm2d(16)  
self.conv3 = nn.Conv2d(16, 16, 3)  
self.bn3   = nn.BatchNorm2d(16)  
self.pool1 = nn.MaxPool2d(2, 2)      
self.conv4 = nn.Conv2d(16, 16, 3)   
self.bn4   = nn.BatchNorm2d(16)
self.conv5 = nn.Conv2d(16, 32, 3)   
self.bn5   = nn.BatchNorm2d(32)  
self.conv6 = nn.Conv2d(32, 16, 3)    
self.bn6   = nn.BatchNorm2d(16)
self.conv7 = nn.Conv2d(16, 10, 5)   
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

Total params: 19,370  
Trainable params: 19,370

## Test Accuracy
Reached 99.49% test accuracy in 15th epoch

## Conclusion
Batch Normalisation after each layer and Data Augmentation (RandomRotation, RandomAffine, ColorJitter) helped to achieve the test accuracy. To reduce the parameters FC layers have not been used.
