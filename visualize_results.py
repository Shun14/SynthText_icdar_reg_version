#! /usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from common import *



def viz_textbb(text_im, wordBB,txt,alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    plt.hold(True)
    W,H = text_im.size

    # plot the character-BB:
    # for i in xrange(len(charBB_list)):
    #     bbs = charBB_list[i]
    #     ni = bbs.shape[-1]
    #     for j in xrange(ni):
    #         bb = bbs[:,:,j]
    #         bb = np.c_[bb,bb[:,0]]
    #         plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    
    print ('wordBB:',wordBB.shape)
    for i in xrange(wordBB.shape[0]):
        bb = wordBB[i]
        bb = np.c_[bb,bb[:,0]]
        print(bb)
        print()
        print(txt[i].decode('utf-8'))
        # plt.text(float(bb[0,:]), float(bb[1,:]), txt[i].decode('utf-8'))
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)

        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in xrange(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=True)

def main(datadir, name):
    file_path = os.path.join(datadir,'total_img',name+'.jpg')
    print(file_path)
    text_im = Image.open(file_path)
    print(text_im.size)
    text_path = os.path.join(datadir,'total_txt', name+'.txt')
    wordBB = []
    txt = []
    with open(text_path,'r') as f:
        for l in f.readlines():
            a = l.strip().split(' ')
            gt_text = a[-1]
            points = a[0].split(',')
            xs = [float(points[i]) for i in [0,2,4,6]]
            ys = [float(points[i]) for i in [1,3,5,7]]
            bbox = []
            txt.append(gt_text)
            bbox.append(xs)
            bbox.append(ys)
            gt_bbox = np.array(bbox)
            bbox_t = np.array(a[0].split(',')).reshape(4,2)
            # print(np.r_[bbox_t[:,0], bbox_t[:,1]])
            
            print('gt_box:',gt_bbox.shape)
            wordBB.append(gt_bbox)
    viz_textbb(text_im,np.array(wordBB), txt)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize the results')
    parser.add_argument('--datadir', default='icpr_data_vertical_1/', type=str)
    parser.add_argument('--name', default='img_5_2', type=str)
    args = parser.parse_args()
    main(args.datadir,args.name)

