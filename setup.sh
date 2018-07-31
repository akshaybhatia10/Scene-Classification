#!/usr/bin/env bash

mkdir tensorboard_dir models features 
wget https://www.dropbox.com/s/yxj6x9ko09a1a98/dataset.tar.gz?dl=0
tar -xzf dataset.tar.gz

pip install numpy
pip install tensorflow
pip install tensorflow-hub