#!/bin/bash

echo "Ready to build."
echo "Press any key to continue or Ctrl+C to exit ..."
#read -n 1

docker build -t gridtools4py .
