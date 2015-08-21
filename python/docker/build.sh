#!/bin/bash

echo "Ready to build container image."
echo "Press any key to continue or Ctrl+C to exit ..."
read -n 1

docker build --rm -t gridtools4py .
