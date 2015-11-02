#!/bin/bash

echo "This container image builds on top of the official NVIDIA CUDA container"
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1

docker build --rm -t gridtools4py .
