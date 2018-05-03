from __future__ import division
#!/usr/bin/python
# encoding: utf-8

import cv2
import math
import os
import  os.path as osp

from shutil import copyfile
import random
import numpy as np
import codecs
from PIL import Image
import argparse

def general_crop(image, tile, reverse_tile=False, margin_ratio=None):
    """Crop the image giving a tile.
    Note: 
    Args:
        image: Image to be crop, [h, w, c].
        tile: [p_0, p_1, p_2, p_3] (clockwise).

    Returns:
        cropped: Patch corresponding to the tile.

    Raises:
        ZeroDivisionError: x[1] == x[0] or x[2] == x[3].
    """
    if reverse_tile:
        tile[1:] = tile[::-1][:3]  
    x = [p[0] for p in tile]
    y = [p[1] for p in tile]
    # phase1:shift the center of patch to image center
    x_center = int(round(sum(x) / 4))
    y_center = int(round(sum(y) / 4))
    im_center = [int(round(coord / 2)) for coord in image.shape[:2]]
    shift = [im_center[0] - y_center, im_center[1] - x_center]
    M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    height, width = image.shape[:2]
    im_shift = cv2.warpAffine(image, M, (width, height))

    # phase2:imrote the im_shift to regular the box
    bb_width = (math.sqrt((y[1] - y[0]) ** 2 + (x[1] - x[0]) ** 2) +
                math.sqrt((y[3] - y[2]) ** 2 + (x[3] - x[2]) ** 2)) / 2
    bb_height = (math.sqrt((y[3] - y[0]) ** 2 + (x[3] - x[0]) ** 2) +
                 math.sqrt((y[2] - y[1]) ** 2 + (x[2] - x[1]) ** 2)) / 2
    if bb_width > bb_height:  # main direction is horizental
        tan = ((y[1] - y[0]) / float(x[1] - x[0] + 1e-8) +
               (y[2] - y[3]) / float(x[2] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180
    else:  # main direction is vertical
        tan = ((y[1] - y[2]) / float(x[1] - x[2] + 1e-8) +
               (y[0] - y[3]) / float(x[0] - x[3] + 1e-8)) / 2
        # degree = 90 + math.atan(tan) / math.pi * 180
        degree = math.atan(tan) / math.pi * 180 - np.sign(tan) * 90
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), degree, 1)
    im_rotate = cv2.warpAffine(im_shift, rotation_matrix, (width, height))
    # phase3:crop the box out.
    x_min = im_center[1] - int(round(bb_width / 2))
    x_max = im_center[1] + int(round(bb_width / 2))
    y_min = im_center[0] - int(round(bb_height / 2))
    y_max = im_center[0] + int(round(bb_height / 2))
    # phase4: add some margin
    if margin_ratio is not None:
        margin_x = int(round((x_max - x_min) * margin_ratio / 2))
        margin_y = int(round((y_max - y_min) * margin_ratio / 2))
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(width, x_max + margin_x)
        y_max = min(height, y_max + margin_y)
    return im_rotate[y_min:y_max, x_min:x_max, :]



def test_icpr_crop():
    image_path = './train_1000/image_1000/TB1..FLLXXXXXbCXpXXunYpLFXX.jpg'
    txt_path = './train_1000/txt_1000/TB1..FLLXXXXXbCXpXXunYpLFXX.txt'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(txt_path) as fo:
        for crop_id, line in enumerate(fo):
            tags = line.strip().split(',')
            points = [float(x) for x in tags[:8]]
            label = ','.join(tags[8:])
            if label == '###' or label == '':
                continue
            xs = [points[i] for i in [0, 2, 4, 6]]
            ys = [points[i] for i in [1, 3, 5, 7]]
            tile = [(x, y) for (x, y) in zip(xs, ys)]
            crop_image = general_crop(image, tile)
            crop_image_path = '{}.jpg'.format(crop_id)
            cv2.imwrite(crop_image_path, crop_image)

def get_icpr_crop_data(datadir):
    root_dir = './' + datadir
    image_dir = os.path.join(root_dir, 'total_img') 
    txt_dir = os.path.join(root_dir, 'total_txt')

    print image_dir
    print txt_dir
    assert os.listdir(image_dir) != os.listdir(txt_dir) , 'error: img list != txt list'
    assert os.listdir(image_dir) != 0 , 'no image'
    total_img_list = os.listdir(image_dir)
    total_num = len(total_img_list)
    train_num = int(total_num/10 * 9)
    test_num = total_num - train_num
    print('total_num:%d, train_num:%d, test_num:%d' % (total_num, train_num, test_num))

    crop_train_root = osp.join(root_dir, 'crop_train_{}/'.format(train_num))
    print('crop_train_root:',crop_train_root)

    crop_test_root = osp.join(root_dir, 'crop_test_{}/'.format(test_num))
    print('crop_test_root:',crop_test_root)
    
    tags_train_file = root_dir+ '/train_{}.tags'.format(train_num)
    
    tags_test_file = root_dir+ '/test_{}.tags'.format(test_num)

    print (tags_train_file, tags_test_file)
    if not osp.exists(crop_train_root):
        os.makedirs(crop_train_root)
    if not osp.exists(crop_test_root):
        os.makedirs(crop_test_root)

    tags_train_fo = open(tags_train_file, 'w')
    tags_test_fo = open(tags_test_file, 'w')
    error_msg = open(root_dir+ '/error.log', 'w')
    error_num = 0
    total_crop_num = 0
    for image_name in total_img_list:
        image_path = osp.join(image_dir, image_name)
        image_id = '.'.join(image_name.split('.')[:-1])
        txt_name = image_id + '.txt'
        txt_path = os.path.join(txt_dir, txt_name)

        if total_img_list.index(image_name) < train_num:
            crop_save_dir = os.path.join(crop_train_root, image_id)
        else :
            crop_save_dir = os.path.join(crop_test_root, image_id)
        
        if not osp.exists(crop_save_dir):
            os.makedirs(crop_save_dir)
        
        #
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            image = Image.open(image_path).convert('RGB')
            image.save(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        assert image is not None, 'image is none! {}'.format(image_path)

        try:
            with open(txt_path) as fo:
                image_crop_id = 0
                for line in fo:
                    tspace = line.strip().split(' ')
                    
                    ts = tspace[0].split(',')
                    points = [float(x) for x in ts[:8]]
                    label = tspace[1]

                    if label == '###' or label == '':
                        continue

                    xs = [points[i] for i in [0, 2, 4, 6]]
                    ys = [points[i] for i in [1, 3, 5, 7]]
                    tile = [(x, y) for (x, y) in zip(xs, ys)]
                    crop_image = general_crop(image, tile)
                    crop_image_path = os.path.join(crop_save_dir, '{}.jpg'.format(image_crop_id))
                    image_crop_id += 1
                    cv2.imwrite(crop_image_path, crop_image)
                    print(crop_image_path)
                    total_crop_num += 0
                    if total_img_list.index(image_name) < train_num:
                        tags_train_fo.write('{} {}\n'.format(crop_image_path, label))
                    else:
                        tags_test_fo.write('{} {}\n'.format(crop_image_path, label))
        except IOError:
            error_msg.write('txt:{}, image:{}\n'.format(txt_path, image_path))
            error_num += 1
        
                
    tags_train_fo.close()
    tags_test_fo.close()
    print('succed:', total_num - error_num)
    print('crop_total_num:',total_crop_num)
    print('finished!!')

if __name__ == "__main__":
    # test_rctw_crop()
    get_icpr_crop_data('icpr_data_1')

