#-*- coding:utf-8 -*-
from PIL import Image,ImageDraw,ImageFont,ImageOps
import PIL
import numpy as np
import random
import os.path as osp
import os
import codecs
import sys
reload(sys) 
sys.setdefaultencoding('utf-8')

data_dir = 'data'
fn=osp.join(data_dir,'newsgroup', 'new/')
txt=[]

files= os.listdir(fn)

print files
random.shuffle(files)
filecnt=3
for filename in files:
    filecnt-=1
    if filecnt==0:
        break
    fc=filename.decode('utf-8')
    fc=fn+fc
    print fc
    with codecs.open(fc,'r') as f:
        for l in f.readlines():
            line=l.strip()
            line=line.decode('utf-8')
            line= line.split()

            #print line
            txt += line
random.shuffle(txt)

print(len(txt))

class LetterImage():
    
    def __init__(self,fontFile='',imgSize=(0,0),imgMode='RGB',bg_color=(0,0,0),fg_color=(255,255,255),fontsize=20):
        self.imgSize = imgSize
        self.imgMode = imgMode
        self.fontsize = 0
        print bg_color
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
        self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]

        fn=osp.join(data_dir,'newsgroup', 'new/')
        self.txt=txt

        
        # if ''==fontFile:
        #     font_path  =random.choice(self.fonts)
        #     self.font = ImageFont.truetype(font_path)
        # else:
        #     self.font = ImageFont.truetype(fontFile,fontsize)


    def GenLetterImage(self,letters, bg_color, fg_color):
        '''Generate the Image of letters''' 
        self.letters = letters
        font_path  =random.choice(self.fonts)
        self.fontsize = random.randint(10,70)
        self.font = ImageFont.truetype(font_path, self.fontsize)
        self.bg_color = bg_color
        self.fg_color = fg_color
        

        (self.letterWidth,self.letterHeight) = self.font.getsize(letters)
        if self.imgSize==(0,0):
            self.imgSize=(int(self.letterWidth + 3),int(self.letterHeight + 4))
        self.imgWidth,self.imgHeight=self.imgSize
        self.img = Image.new(self.imgMode, self.imgSize, self.bg_color)
        self.imgSize = (0,0)
        self.drawBrush = ImageDraw.Draw(self.img)
        textY0 = (self.imgHeight-self.letterHeight+1)/2
        textY0 = int(textY0)
        textX0 = int((self.imgWidth-self.letterWidth+1)/2)
        self.drawBrush.text((textX0,textY0), self.letters, fill=self.fg_color,font=self.font)

    
    def SaveImg(self,saveName=''):
        if ''==saveName.strip():
            saveName = str(self.letters.encode('utf-8'))+'.jpeg'
        fileName,file_format = saveName.split('.')
        fileName+='_'+str(self.fontsize)+'.'+file_format
        # print fileName,file_format
        try:
            self.img.save('horizontal_2/' +fileName, file_format)
        except IOError:
            PIL.ImageFile.MAXBLOCK = self.img.size[0] * self.img.size[1]
            self.img.save('horizontal_2/' +fileName, file_format)
        
        return './horizontal_2/'+fileName
    
    def Show(self):
        self.img.show()

def clearpictures():
    import os
    png = os.listdir(os.curdir)
    for i in png:
        if os.path.splitext(i)[1]==".png":
            os.remove(i)
            

if __name__=='__main__':
    letterList = []

    # letterList.append(LetterImage(bg_color=(0,0,255),fontsize=50))
    # letterList.append(LetterImage(fontFile='',bg_color=(0,0,255),fontsize=30))
    
    num_letter = 1000000
    # num_letter = 20
    import cv2
    
    npareiImg =[]

    bg_fg = {}
    bg_color_list  = []
    try:
        with codecs.open('img_shelter.txt', 'r') as fin:
            lines = fin.readlines()
            for l in lines:
                img_rgb = l.strip().split()[0][1:]
                r = int(img_rgb[0:2], 16)
                g = int(img_rgb[2:4], 16)
                b = int(img_rgb[4:], 16)
                font_rgb = l.strip().split()[1][1:]
                font_r = int(font_rgb[0:2], 16)
                font_g = int(font_rgb[2:4], 16)
                font_b = int(font_rgb[4:], 16)
                # key = '{}{}{}'.format(r,g,b)
                key = img_rgb
                bg_color_list.append(key)
                bg_fg[key] = (font_r, font_g, font_b)
    except:
        print('error')

    all_list = []
    letter = LetterImage()
    print(len(bg_color_list))
    for i in range(num_letter):
        bg_color= bg_color_list[random.randint(0, 19)]
        r = int(bg_color[0:2], 16)
        g = int(bg_color[2:4], 16)
        b = int(bg_color[4:], 16)
        print('{}/{}'.format(i, num_letter))
        (font_r, font_g, font_b) = bg_fg[bg_color]
        j = i%len(txt)
   
        letter.GenLetterImage(txt[j].strip(), bg_color=(r,g,b), fg_color=(font_r, font_g, font_b))
        img_path = letter.SaveImg('{}.jpeg'.format(i))
        all_list.append('{},{}\n'.format(img_path, txt[j].encode('utf-8').decode('utf-8')))


    with codecs.open('horizontal_2.tags', 'w') as fout:
        fout.writelines(all_list)
    # cv2.waitKey()
