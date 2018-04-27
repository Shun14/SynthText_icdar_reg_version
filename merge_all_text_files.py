# -*- coding: utf-8 -*- 
import os

def main():
    rootDir = '5000Data/txt'
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
    merge_text_tags = open('5000Data/merge_text.tags', 'w')
    merge_text_tags.writelines(all_list)
    merge_text_tags.close()

if __name__=='__main__':
    main()


