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
import pickle
## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 20# no. of times to use the same image
SECS_PER_IMG = 10 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText_cartoon_viz.h5'

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
def add_res_to_cp(imname, res, out_dir):
  cp_path = os.path.join(out_dir,'cp')
  if not os.path.exists(cp_path):
    os.makedirs(cp_path) 

  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imname, i)
    del res[i]['img']
    save_path = os.path.join(cp_path, dname)+'.pickle'
    with open(save_path, 'wb') as f:
      pickle.dump(res[i], f, protocol=pickle.HIGHEST_PROTOCOL)
    
  

def save_res_to_file(imgname,res, filepath='icdar_3_data'):
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
  out_dir = args.output_dir
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
  idict = {'name':'cp', 'res':None}
  res_list = []
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
        res = RV3.render_text(img,depth,seg,area,label, imname,data_dir=out_dir,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
        
        t2=time.time()
        if len(res) > 0:
          #TODO multi thread
          # add_res_to_db(imname, res, out_db)
          save_res_to_file(imname, res, out_dir)
          # for x in xrange(len(res)):
          #   del res[x]['img']
          #
          # res_list.append(res)
          # add_res_to_cp(imname, res, out_dir)

        print '*********time consume in each pic',(t2-t1)/INSTANCE_PER_IMAGE
        print ('img length:', i/(end-start))
        if viz:
          if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
            break
    except:
        traceback.print_exc()
        print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
        continue
  #out_db.close()
  idict['res'] = res_list
  cp_path = os.path.join(out_dir,'cp')
  if not os.path.exists(cp_path):
    os.makedirs(cp_path)
  #
  # with open('total.pickle', 'wb') as f:
  #   pickle.dump(idict, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    # __range = '%d,%d' %(2 * i + 1, 2*(i+1))
    # __range = '1,3'
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    # parser.add_argument('--multi', default='yes', type=str)
    parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations') 
    parser.add_argument('--range',default=__range,type=str)
    parser.add_argument('--output_dir',default = 'icpr_data_vertical_new_1',type=str)
    args = parser.parse_args()
    p.apply_async(main1, args=(args,))
  # main1(args)
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
