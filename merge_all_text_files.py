# -*- coding: utf-8 -*- 
import os
import os.path as osp
def main(args):
    
    rootDir = os.path.join(args.datadir, 'txt')
    dir_list = os.listdir(rootDir)
    all_list = []
    for lists in dir_list:
        path = os.path.join(rootDir, lists)
        # print path
        for txt in os.listdir(path):
            txt_path = os.path.join(path, txt)
            # print txt_path
            file = open(txt_path, 'r')
            lines = file.readlines()
            all_list += lines

    print 'all list:',len(all_list)
    merge_text_tags = open( osp.join(args.datadir, 'merge_text.tags'), 'w')
    merge_text_tags.writelines(all_list)
    merge_text_tags.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge all texts to one tags')
    parser.add_argument('--datadir', default='10000_img_output_data', type=str)
    args = parser.parse_args()
    main(args)


