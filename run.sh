#!/bin/bash 
echo 'start time'
start=$(date +%s)
python gen_cartoon_viz.py --output_dir=$1
cp -r $1 /home/zhishengzou/datasets/ICPR
python gen_crop.py --name=$1

echo 'end time'
end=$(date +%s)

echo $(( $end - $start ))
