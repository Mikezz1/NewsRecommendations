wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
unzip MINDsmall_train.zip -d data/MINDsmall_train
rm  MINDsmall_train.zip

wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_dev.zip -d data/MINDsmall_dev
rm MINDsmall_dev.zip

git clone https://github.com/RUCAIBox/RecDatasets
cd RecDatasets/conversion_tools
pip install -r requirements.txt

cd ../../

python RecDatasets/conversion_tools/run.py --dataset mind_small_train \
--input_path data/MINDsmall_train --output_path dataset/mind_small_train \
--convert_inter  

python RecDatasets/conversion_tools/run.py --dataset mind_small_dev \
--input_path data/MINDsmall_dev --output_path dataset/mind_small_dev \
--convert_inter 
