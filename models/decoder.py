#Face Decoder Class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import numpy as np

class Decoder(nn.Module):

    def __init__(self, batchSize):
        super(Decoder, self).__init__()
      

        #Multi-Layer Perceptron Layer
        self.mlp = nn.Sequential(nn.Linear(4096,1000), nn.ReLU(), nn.Linear(1000,50176), nn.ReLU()) #14x14x256
        
        #Texture generator
        self.texture_gen = nn.Sequential(
             #14x14x256
            nn.ConvTranspose2d(256, 128, 6, stride=2,padding=2),
        
            nn.ReLU(),
            
            #28x28x128
            nn.ConvTranspose2d(128, 64, 6, stride=2,padding=2),
           
            nn.ReLU(),

            #56x56x64
            nn.ConvTranspose2d(64, 32, 6, stride=2,padding=2),

            nn.ReLU(),

            #112x112x32
            nn.ConvTranspose2d(32, 32, 6, stride=2,padding=2),
         
            nn.ReLU(),

            #224x224x32
            nn.Conv2d(32,3,1,stride=1),nn.Sigmoid()

            #224x224x3
        )
        self.batch_size=batchSize




    def forward(self, x):
        

        x=self.mlp(x)
   
        x=x.view(self.batch_size,256,14,14)
        texture=self.texture_gen(x)

        return texture


    def forward_test(self, x):
        
  
        x=self.mlp(x)
        x=x.view(1,256,14,14)
        texture=self.texture_gen(x)

        np_texture = texture.cpu().detach().numpy()

        tx=np.moveaxis(np_texture[0], 0, -1) *255.0
        tx=np.rint(tx)
        tx=tx.astype('uint8')
            

        return  tx



    def test(self,x,encoder):

        x=torch.Tensor(x)
        x = x.permute(2,0,1).view(1, 3, 224, 224)
        x -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1) #standardize
        x=encoder(x)

        x=self.mlp(x.type(torch.float))
     
        x=x.view(1,256,14,14)
        texture=self.texture_gen(x)

        np_texture = texture.cpu().detach().numpy()

        tx=np.moveaxis(np_texture[0], 0, -1) *255.0
        tx=np.rint(tx)
        tx=tx.astype('uint8')
            


        return  tx