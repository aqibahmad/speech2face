import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VoiceEncoder(nn.Module):

    def __init__(self, batchSize):
        super(VoiceEncoder, self).__init__()
      
        #Input tensor (spectogram) shape => (2, 598, 257)
        self.batch_size = batchSize


        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(128, 128, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(128, 128, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(128, 256, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, 4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 4, stride=2),
          
            nn.AvgPool2d((6, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            
        ) 
        
        self.linear = nn.Sequential(
          

            nn.Linear(512 * 1 * 57 , 4096), 
            nn.ReLU(),
            nn.Linear(4096,4096), 

            

        )




    def forward(self, x):
        
        x = self.conv(x)
       
        x = x.view(x.size(0), -1)


        x=self.linear(x)
        vgg_4096_vector = F.normalize(x, p=2, dim=1)

        return vgg_4096_vector


    



