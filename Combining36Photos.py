imCount=0
ind=0
t= glob.glob("F:/vggface2_train/train/*")
p = glob.glob(t[ind]+"/*.jpg")
check1=True
imCount = len(p)-9
while (ind < len(p)):
    names = p[imCount:imCount+36]
    if (len(names) == 36):
        imCount+=36
    elif (ind < len(t)-1):
        ind+=1
        p = glob.glob(t[ind]+"/*.jpg")
        imCount = 36-len(names)
        names += p[:imCount]
    else:
        check1=False
        if (len(names) == 0):
            break
    images = [Image.open(x) for x in names]
    
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
    
    file_name = 'testImageavc.jpg'
    new_im.save(file_name)