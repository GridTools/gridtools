#!/bin/bash

echo "About to build a container with a GPU backend for GridTools."
echo "This container image builds on top of the official NVIDIA CUDA container 'cuda:latest'"
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1
docker build --rm --tag=gridtools4py:cpu --file=Dockerfile.gpu .
