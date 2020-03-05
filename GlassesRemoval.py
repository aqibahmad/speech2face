import glob
import PIL
from PIL import Image
from google.cloud import vision
import shutil, os
import io
import os.path
from os import path

###Detect if Person is wearing glasses.
imCount=45280
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Taimoor\\Desktop\\speech2face-f9478dbdd803.json"

while (imCount < len(p)):
    names = p[imCount:imCount+16]
    imCount+=16
    images = []
    for x in names:
        temp = Image.open(x)
        images.append(temp.copy())
        temp.close()
    new_im = Image.new('RGB', (224*4,224*4))
    x_offset = 0
    i = 0
    y=-224
    for im in images:
        if (i % 4 == 0):
            y+=224
            x_offset=0
        new_im.paste(im, (x_offset,y))
        x_offset += 224
        i+=1
    
    file_name = 'testImage.jpg'
    new_im.save(file_name)
    client = vision.ImageAnnotatorClient()
    #image = vision.types.Image()

    #image.source.image = 
    #file_name = 'n000045_0056_01.jpg'

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.object_localization(image = image)

    for obj in response.localized_object_annotations:
        #print (obj)
        if ("glasses" in obj.name.lower()):
            x1 = obj.bounding_poly.normalized_vertices[0].x*896
            y1 = obj.bounding_poly.normalized_vertices[0].y*896
            x2 = obj.bounding_poly.normalized_vertices[2].x*896
            y2 = obj.bounding_poly.normalized_vertices[2].y*896
            midX = (x1+x2)/2
            midY = (y1+y2)/2
            index = int(midY//224 ) *4 + int(midX//224 )
            if (path.exists(names[index])):
                shutil.move(names[index],'GlassesRejected')
            print (names[index])
    os.remove(file_name)