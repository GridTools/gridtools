#!/bin/bash

echo "About to build a container with a GPU backend for GridTools."
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1
docker build --rm --tag=gridtools4py:gpu --file=Dockerfile .
