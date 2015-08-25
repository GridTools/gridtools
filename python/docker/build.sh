#!/bin/bash

echo "The container image will use the NVIDIA driver version 352.21"
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1

docker build --rm -t gridtools4py .
