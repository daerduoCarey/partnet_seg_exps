# This is for Tensorflow 1.12

/usr/local/cuda-9.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC="/home/kaichun/venvs/py3.6/lib64/python3.6/site-packages/tensorflow/include/"
TF_LIB="/home/kaichun/venvs/py3.6/lib64/python3.6/site-packages/tensorflow/"

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-9.0/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

