import glob
import time
import numpy as np
import cv2
import shutil, os
import threading
import math

def Morph(src, dest):
    for i in range(len(dest)):
        tr = src[-11:-4]+"_"+dest[i][-11:-4]+'.png'
        if (os.path.isfile("Morphed\\"+tr)):
            continue
        !python face_morpher-dlib/facemorpher/morpher.py --src={src} --dest={dest[i]} --height=224 --width=224 --out_frames=face_morpher-dlib/examples --plot --background=average
        shutil.move(tr,"Morphed")
        #os.rename(r'Morphed\frame001.png',tr)

def ReadFile(name):
    f = open(name,"r")    
    lines = f.readlines()
    L = []
    for line in lines:
        temp = line.split(',')
        L.append([int(temp[0]),int(temp[1][:-1])])
    f.close()
    return np.array(L)

if not os.path.exists("Morphed"):
    os.mkdir("Morphed")
Landmarks = glob.glob("Averaged1/*.txt")
Textures = np.array(glob.glob("Averaged1/*.png"))

### Set These ###

start = 4645
end = 4872
cores = 4

num_threads = math.ceil(150/(cores*2))
start_time = time.time()
for i in range(start, end):
    if (i == len(Textures)):
        break
        
    with open("filenum.txt","a+") as f:
        f.write(str(i)+'\n')
    LM1 = ReadFile(Landmarks[i])
    tex1 = cv2.imread(Textures[i])    
    
    Distance = np.zeros(len(Landmarks))
    for j in range(len(Landmarks)):
        if i == j:
            Distance[j] = float('inf')
            continue
        LM2 = ReadFile(Landmarks[j])
        tex2 = cv2.imread(Textures[j])
        Distance[j] =  10.0 * np.linalg.norm(LM1-LM2) + np.linalg.norm(tex1-tex2) 
    Selected = Distance.argsort()[:150]
    print("--- %s seconds ---" % (time.time() - start_time))
    print ("Selected")
    start_time = time.time()
    #print (Textures[Selected])
    threads = []
    pic = 0
    while pic < len(Textures[Selected]):
        x = threading.Thread(target=Morph, args=(Textures[i], Textures[Selected][pic:pic+num_threads],))
        pic += num_threads
        threads.append(x)
        x.start()
    print (len(threads))
    for th in threads:
        th.join()
print("--- %s seconds ---" % (time.time() - start_time))