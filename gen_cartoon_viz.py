# -*- coding: utf-8 -*-
# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import cv2 as cv
import scipy.io as sio
import time
from text_utils import *
import multiprocessing
## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 5# no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText_cartoon_viz.h5'

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print colorize(Color.RED,'Data not found and have problems downloading.',bold=True)
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')

# @profile
def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    print 'type of res[i][\'txt\'] ',type(res[i]['txt'])
    print 'name', dname 
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    db['data'][dname].attrs.create('txt', res[i]['txt'], dtype=h5py.special_dtype(vlen=unicode))
    print 'type of db ',type(db['data'][dname].attrs['txt']) 
    print 'successfully added'
    #print res[i]['txt']
    # print res[i]['img'].shape
    #print 'charBB',res[i]['charBB'].shape
    #print 'charBB',res[i]['charBB']
    #print 'wordBB',res[i]['wordBB'].shape
    #print 'wordBB',res[i]['wordBB']
    '''
    img = Image.fromarray(res[i]['img'])
    hsv_img=np.array(rgb2hsv(img))
    print 'hsv_img_shape',hsv_img.shape
    print 'hsv_img',hsv_img
    H=hsv_img[:,:,2]
    print 'H_channel',H.shape,H
    #img = Image.fromarray(db['data'][dname][:])
    '''

def save_res_to_file(imgname,res, filepath='5000Data'):
  """
  Add the synthetically generated text image instance
  and other metadata to the file.
  """

  img_path = os.path.join(filepath, 'total_img')
  
  if not os.path.exists(img_path):
    os.makedirs(img_path)
  
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    wordBB = res[i]['wordBB']
    img = res[i]['img']
    imgpath = os.path.join(img_path, dname)+ '.jpg'
    cv.imwrite(imgpath,cv.cvtColor(img, cv.cv.CV_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def rgb2hsv(image):
    return image.convert('HSV')

def rgb2gray(image):
    
    rgb=np.array(image)
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def parse_txt(txt_name):
  with open(txt_name) as f:
    tmp=f.readlines()
  tmp=[t.strip() for t in tmp]
  tmp=[t.replace('\xef\xbb\xbf','') for t in tmp]
  tmp=[t.split(',')[0:8] for t in tmp]
  box=[]
  for t in tmp:
    x=[int(l) for l in t[0::2]]
    y=[int(m) for m in t[1::2]]
    box.append([min(x),min(y),max(x),max(y)])
  return box

# @profile
def main1(args):
  viz = args.viz
  ranges = args.range
# def main1(viz=False,ranges='0,100'):
  
  # OUT_FILE = 'results/icdar_%s_%s.h5'%(ranges.split(',')[0],ranges.split(',')[1])

  # open the output h5 file:
  # out_db = h5py.File(OUT_FILE,'w')
  # out_db.create_group('/data')
  # print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  print(ranges)
  ranges=ranges.split(',')
  start=int(ranges[0])
  end=int(ranges[1])
  for i in range(start, end):
    t1 = time.time()
    try:
        imname = 'img_%d'%i
        print 'imname: %s' %imname
        img = Image.open('./raw_data/img/' + imname + '.jpg')
        box=parse_txt('./raw_data/train_gts/gt_'+imname+'.txt')
        depth = sio.loadmat('./raw_data/depth_output/img_%d/predict_depth.mat'%i)['data_obj']
        seg = sio.loadmat('./raw_data/seg_output/img_%d_seg.mat'%i)['seg_mat']
        areamat = sio.loadmat('./raw_data/seg_output/img_%d_area.mat'%i)['area_mat']
        areamat = np.array(areamat.reshape((areamat.shape[0], )))
        label = np.arange(1, areamat.shape[0] + 1)
        area = [areamat[q][0][0][0] for q in range(areamat.shape[0])]
        area = np.array(area)

        sz = depth.shape[:2][::-1]

        w, h = img.size
        if w!=sz[0]:
          for j in range(len(box)):
            box[j][0]=box[j][0]*sz[0]*1./w
            box[j][1]=box[j][1]*sz[1]*1./h
            box[j][2]=box[j][2]*sz[0]*1./w
            box[j][3]=box[j][3]*sz[1]*1./h
        
        img = np.array(img.resize(sz,Image.ANTIALIAS))
        seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))
        print colorize(Color.RED,'%d of %d'%(i,end), bold=True)
        res = RV3.render_text(img,depth,seg,area,label,imname,data_dir='5000Data',
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
        t2=time.time()
        if len(res) > 0:
          #TODO multi thread
          # add_res_to_db(imname, res, out_db)
          save_res_to_file(imname, res)
        print '*********time consume in each pic',(t2-t1)/INSTANCE_PER_IMAGE
        print ('img length:', i/(end-start))
        if viz:
          if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
            break
    except:
        traceback.print_exc()
        print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
        continue
  out_db.close()

   
def main(viz=False):
  # open databases:
  print colorize(Color.BLUE,'getting data..',bold=True)
  db = get_data()
  print colorize(Color.BLUE,'\t-> done',bold=True)

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  
  for i in xrange(start_idx,end_idx):
    t1=time.time()
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      img_resize=img.resize(db['depth'][imname].shape[1:3])
      depth = db['depth'][imname][:].T
      print 'depth shape,img shape',depth.shape,np.array(img).shape
      print 'depth info',depth
      print 'depth max min',np.max(depth),np.min(depth)
      #depth = depth[:,:,1]
      #modify the depth with HSV H_channel
      
      #img_resize=img.resize(depth.shape)
      hsv_img=np.array(rgb2hsv(img_resize))
      print 'hsv_img_shape',hsv_img.shape
      #print 'hsv_img',hsv_img
      H=hsv_img[:,:,2]
      H=H.T
      H=H.astype('float32')
      print 'H_channel',H.shape,H 
      print 'H_max min',np.max(H),np.min(H)
      print 'scale',np.max(depth)/np.max(H)
      #depth= (np.max(depth)/np.max(H))*H
      #depth= H
      #print np.isnan(H).any()
      #print np.isinf(H).any()
      #print np.isnan(depth).any()
      #print np.isinf(depth).any()
      print 'depth shape',depth.shape
      #print 'depth info',depth
      print 'depth max min',np.max(depth),np.min(depth)
      
      gray=np.array(rgb2gray(img_resize))
      #print 'gray',gray.shape,gray
      depth= (np.max(depth)/np.max(gray))*gray.astype('float32')
      #add more blur 
      #mean blur 
      kernel = np.ones((5,5),np.float32)/25
      gray = cv2.filter2D(gray,-1,kernel)
      #print 'gray',gray.shape,gray
      
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']
      
      print 'seg info',seg.shape,area.shape,label.shape
      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      t2=time.time()
      
      
      for ct in range(5):
      
        if len(res) > 0:  
            # non-empty : successful in placing text:
            add_res_to_db(imname,res,out_db)
            break
        else:
            res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      print 'time consume in each pic',(t2-t1)/INSTANCE_PER_IMAGE
      # visualize the output:
      if viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  
  
  argsList = []
  #multipy
  startTime = time.time()
  
  # if args.multi == 'yes':
  print 'Parent process %s' % os.getpid()
  p = multiprocessing.Pool()
  # parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  
  for i in range(0, 10):
    __range = '%d,%d' %(100 * i + 1, 100*(i+1))
    #__range = '%d,%d' %(2 * i + 1, 2*(i+1))
  #  __range = '1,3'
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    # parser.add_argument('--multi', default='yes', type=str)
    parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations') 
    parser.add_argument('--range',default=__range,type=str)
    args = parser.parse_args()
    p.apply_async(main1, args=(args,))
  print 'waiting for all done'
  p.close()
  p.join()
  # elif args.multi == 'no':
  #   parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  #   parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  #   parser.add_argument('--range',default='101,102',type=str)

  #   args = parser.parse_args()
  #   main1(args)

  # else:
  #     print 'error, no muti'
  
  endTime = time.time()
  allTime = endTime - startTime
  print 'allTime:',allTime
  print 'All done'
