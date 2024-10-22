import os
import h5py
from common import *
import sys,time

def main(args):
  dataDir = os.path.join(args.datadir,'total_img')
  total_num = args.total_num
  allCount = 0
  unit_time = 30.0
  while allCount < total_num :
    list = os.listdir(dataDir)
    new_count = len(list)
    speed = (new_count - allCount)/unit_time
    allCount = new_count
    print ('speed:', speed, 'pic/s')
    print ('need time:', (total_num-allCount)*2.0/3600.0, 'hours')
    print "total number of images : ", colorize(Color.RED, allCount , highlight=True)
    time.sleep(unit_time) 
if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Count the results')
  parser.add_argument('--datadir', default='output_data', type=str)
  parser.add_argument('--total_num', default=20000, type=int)
  args = parser.parse_args()
  main(args)

