### To Select Front Facing Neutral Expression images###
import glob
import os
from google.cloud import vision
import io
import shutil
import PIL
from PIL import Image

#x= glob.glob("*")
#print (os.path.isfile(x[8]))


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Taimoor\\Desktop\\speech2face-f9478dbdd803.json"



ind=0
t= glob.glob("F:/vggface2_train/train/*")
p = glob.glob(t[ind]+"/*.jpg")
check1=True
imCount=0
the_file = open('FolderIndex.txt', 'a')
the_file.write(t[0]+"\n")
the_file.close()
while (True):
    names = p[imCount:imCount+36]
    if (len(names) == 36):
        imCount+=36
    elif (ind < len(t)-1):
        print("Changing Folder")
        ind+=1
        the_file = open('FolderIndex.txt', 'a')
        the_file.write(t[ind]+",  "+str(ind)+"\n")
        the_file.close()
        p = glob.glob(t[ind]+"/*.jpg")
        imCount = 36-len(names)
        names += p[:imCount]
    else:
        check1=False
        if (len(names) == 0):
            break
    images = []
    for x in names:
        temp = Image.open(x)
        images.append(temp.copy())
        temp.close()
    images = [im.resize((224,224),PIL.Image.ANTIALIAS) for im in images]
    new_im = Image.new('RGB', (224*6,224*6))
    x_offset = 0
    i = 0
    y=-224
    for im in images:
        if (i % 6 == 0):
            y+=224
            x_offset=0
        new_im.paste(im, (x_offset,y))
        x_offset += 224
        i+=1
    
    file_name = 'API_Results/testImage.jpg'
    new_im.save(file_name)
    
    
    client = vision.ImageAnnotatorClient()
    #image = vision.types.Image()

    #image.source.image = 

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image,max_results=100)
    image_file.close()
    properties = ['surprise_likelihood','joy_likelihood','headwear_likelihood','sorrow_likelihood','anger_likelihood','under_exposed_likelihood','blurred_likelihood']
    for face in response.face_annotations:
        index = -1
        check = True
        for pr in properties:
            temp = getattr(face,pr)
            if (temp != 1 and temp != 2):
                check = False
                break
        if (check == True): #low Likelihoods
            if ( abs(getattr(face,'roll_angle')) > 5 or abs(getattr(face,'pan_angle')) > 5 or abs(getattr(face,'tilt_angle')) > 5 ):
                check = False

        if (check == True):
            vertices = [[v.x,v.y] for v in face.bounding_poly.vertices]
            XmidPoint = (vertices[0][0] + vertices[2][0])/2.0
            YmidPoint = (vertices[0][1] + vertices[2][1])/2.0
            index = int(YmidPoint//224 ) *6 + int(XmidPoint//224 )
            Tempimg = new_im.crop((vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1])) 
            Tempimg=Tempimg.resize((224,224),PIL.Image.ANTIALIAS)
            # Shows the image in image viewer 
            Tempimg.save('API_Results/'+names[index][-19:-12]+"_"+names[index][-11:])
            print(names[index][-19:]+" Detected as frontal neutral expression face. ")
#             types = ["LEFT_EYE","RIGHT_EYE","LEFT_OF_LEFT_EYEBROW","RIGHT_OF_LEFT_EYEBROW","LEFT_OF_RIGHT_EYEBROW","RIGHT_OF_RIGHT_EYEBROW","MIDPOINT_BETWEEN_EYES","NOSE_TIP","UPPER_LIP","LOWER_LIP","MOUTH_LEFT","MOUTH_RIGHT","MOUTH_CENTER","NOSE_BOTTOM_RIGHT","NOSE_BOTTOM_LEFT","NOSE_BOTTOM_CENTER","LEFT_EYE_TOP_BOUNDARY","LEFT_EYE_RIGHT_CORNER","LEFT_EYE_BOTTOM_BOUNDARY","LEFT_EYE_LEFT_CORNER","RIGHT_EYE_TOP_BOUNDARY","RIGHT_EYE_RIGHT_CORNER","RIGHT_EYE_BOTTOM_BOUNDARY","RIGHT_EYE_LEFT_CORNER","LEFT_EYEBROW_UPPER_MIDPOINT","RIGHT_EYEBROW_UPPER_MIDPOINT","LEFT_EAR_TRAGION","RIGHT_EAR_TRAGION","LEFT_EYE_PUPIL","RIGHT_EYE_PUPIL","FOREHEAD_GLABELLA","CHIN_GNATHION","CHIN_LEFT_GONION","CHIN_RIGHT_GONION"]    
#             x= np.array(face.landmarks)
#             for iter1 in x:
#                 with open('results/'+names[index][-19:-12]+"_"+names[index][-11:]+'.txt', 'a') as the_file:
#                     the_file.write(types[iter1.type-1])
#                     the_file.write('\tx: '+str((iter1.position.x%224))+'y: '+str(iter1.position.y%224)+'z: '+str(iter1.position.z)+"\r\n")
    os.remove(file_name)
    
    new_im.close()
    for i in images:
        i.close()

    if (check1 == False):
        break
            