#!/bin/bash

GPU=0 ./nvidia-docker/nvidia-docker run --rm                                     \
                                        -it                                      \
                                        -P                                       \
                                        gridtools4py:gpu
