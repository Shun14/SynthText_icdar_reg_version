# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *



def viz_textbb(text_im, charBB_list, wordBB, txt,alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    plt.hold(True)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    for i in xrange(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1]
        for j in xrange(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    print("*****text len:****", len(txt))
    print ('wordBB:',wordBB.shape[-1])
    for i in xrange(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        print('****count:',i)
        # if len(txt) > i:
        #   plt.text(bb[0,0], bb[1,0], txt[i], size=20)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in xrange(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

def main(db_fname):
    print 'name_db:', db_fname
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print "total number of images : ", colorize(Color.RED, len(dsets), highlight=True)
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        viz_textbb(rgb, [charBB], wordBB,txt)
        print "image name        : ", colorize(Color.RED, k, bold=True)
        print "  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1])
        print "  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1])
        print "  ** text         : ", colorize(Color.GREEN, 'text')
        for l in txt:
          print l

        if 'q' in raw_input("next? ('q' to exit) : "):
            break
    db.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize the results')
    parser.add_argument('--name', default='icdar_101_102.h5', type=str)
    parser.add_argument('--datadir', default='testresults/', type=str)
    args = parser.parse_args()
    main(os.path.join(args.datadir ,args.name ))

