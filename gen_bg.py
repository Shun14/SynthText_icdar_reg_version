#-*- coding:utf-8 -*-
from PIL import Image
import codecs

x = 1280
y = 1280

with codecs.open('img_shelter.txt', 'r') as fin:
    lines = fin.readlines()
    z = 0
    for l in lines:
        img_rgb = l.strip().split()[0][1:]
        print(img_rgb)
        im = Image.new('RGB', (x, y))
        z += 1
        for i in range(0, x):
            for j in range(0, y):
                im.putpixel((i,j) ,(int(img_rgb[0:2], 16), int(img_rgb[2:4], 16), int(img_rgb[4:], 16)))
#        im.show()
#        im.save('img_bg/{}.jpg'.format(z),'jpeg')
