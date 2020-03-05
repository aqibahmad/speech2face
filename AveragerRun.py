import glob
p = glob.glob("results/*")
for name in p:
    !python face_morpher-dlib/facemorpher/averager.py --images={name} --background=average --height=224 --width=224 --out=Averaged/{name[8:]}.png