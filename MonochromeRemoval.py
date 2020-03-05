from PIL import Image,ImageChops
import glob
import shutil, os

def is_MonoChrome(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return False
    return True

t= glob.glob("API_Results/*.jpg")

for pic in t:
    x = Image.open(pic)
    if (is_MonoChrome(x)):
        shutil.move(pic,'Monochrome')
        print (pic)
    x.close()

#
#print(is_greyscale(x))