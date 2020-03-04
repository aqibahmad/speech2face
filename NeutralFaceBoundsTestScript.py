import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open('testImage0.jpg'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((45,481),177-45,635-481,linewidth=1,edgecolor='r',facecolor='none')
rect1 = patches.Rectangle((485,682),634-485,855-682,linewidth=1,edgecolor='r',facecolor='none')
rect2 = patches.Rectangle((485,7),636-485,182-7,linewidth=1,edgecolor='r',facecolor='none')
rect3 = patches.Rectangle((1141,1158),1190-1141,1215-1158,linewidth=1,edgecolor='r',facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
plt.show()
plt.save("av.jpg")

im = Image.open(r"testImage0.jpg") 
  

  
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((1141, 1158, 1190, 1215)) 
im1=im1.resize((224,224))
# Shows the image in image viewer 
im1.save("abc.jpg")