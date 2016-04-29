#!/bin/bash

echo "About to build a container with a CPU-only backend for GridTools."
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1
docker build --rm --tag=gridtools4py:cpu --file=Dockerfile .
