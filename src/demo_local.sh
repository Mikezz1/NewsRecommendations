#!/bin/bash

mode=$1
nGPU=1
model_dir='../model/NRMS'
model='NRMS'
use_category=False
use_subcategory=False
train_data_dir='../../data/MINDsmall_train'
test_data_dir='../../data/MINDsmall_dev'
enable_gpu=False
glove_embedding_path='../../data/glove.6B.300d.txt'
log_steps=10

if [ ${mode} == train ]
then
 epochs=5
 batch_size=32
 lr=0.0003
 user_log_mask=False
 prepare=True
 python -u main.py --mode train --model_dir ${model_dir} --batch_size ${batch_size} --epochs ${epochs} --model ${model} \
 --lr ${lr} --user_log_mask ${user_log_mask} --prepare ${prepare} --nGPU ${nGPU} \
 --use_category ${use_category} --use_subcategory ${use_subcategory} --train_data_dir ${train_data_dir} --test_data_dir ${test_data_dir}\
 --enable_gpu ${enable_gpu} --glove_embedding_path ${glove_embedding_path} --log_steps ${log_steps}
elif [ ${mode} == test ]
then
 user_log_mask=True
 batch_size=128
 load_ckpt_name=$2
 prepare=True
 python -u main.py --mode test --model_dir ${model_dir} --batch_size ${batch_size} --user_log_mask ${user_log_mask} \
 --load_ckpt_name ${load_ckpt_name} --model ${model} --prepare ${prepare} --nGPU ${nGPU} \
 --use_category ${use_category} --use_subcategory ${use_subcategory}
else
 echo "Please select train or test mode."
fi