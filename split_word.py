#!/usr/bin/python3
import os
import random

import codecs
#file_list = os.listdir('data/newsgroup')
file_list = ['sogo_gbk.txt']
#random.shuffle(file_list)
MAX_LINE_NUM = 10
for change_file in file_list:
    if len(change_file.split('.')) == 2:
        change_2_file_list = []
        print('change_file',change_file)
        def random_line(line):
            new_list = []
        
            if len(line) ==0:
                return new_list
            range = random.randint(5, 12)
            if  len(line) > range:
                split_num = random.randint(4, len(line)-1)
                split_part1 = line[:split_num]
                split_part2 = line[split_num:]
                
                return random_line(split_part1) + random_line(split_part2)
            else:
                new_list += [line+'\n']
            return new_list
        
        with codecs.open(os.path.join('data/newsgroup',change_file), 'r', encoding='UTF-8') as f:
            for l in f.readlines():
                line = l.strip()
                line = line.replace(' ','')
                
                # line = line.decode('utf-8')
                new_line_list = []
                
                if len(line) > MAX_LINE_NUM:
                    new_list = random_line(line)
                    new_line_list += new_list
                else:
                    new_line_list.append(line+'\n')
                change_2_file_list += new_line_list
        
        
        
        with codecs.open(os.path.join('data/newsgroup/new',change_file), 'w', encoding='UTF-8') as f:
            f.writelines(change_2_file_list)
        
        print('all finished')
