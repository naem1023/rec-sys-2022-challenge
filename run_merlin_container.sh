docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
    -v ./transfo
    nvcr.io/nvidia/merlin/merlin-pytorch-training:22.03 /bin/bash