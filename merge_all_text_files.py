# -*- coding: utf-8 -*- 
import os
import os.path as osp
def main(args):
    
    rootDir = os.path.join(args.datadir, 'total_txt')
    dir_list = os.listdir(rootDir)
    all_list = []
    for lists in dir_list:
        # print path
        txt_path = osp.join(rootDir,lists)
            # print txt_path
        file = open(txt_path, 'r')
        lines = file.readlines()
        all_list += lines

    print 'all list:',len(all_list)
    merge_text_tags = open( osp.join('.','merge_text.tags'), 'w')
    merge_text_tags.writelines(all_list)
    merge_text_tags.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge all texts to one tags')
    parser.add_argument('--datadir', default='10000_img_output_data', type=str)
    args = parser.parse_args()
    main(args)


