# NVIDIA Docker utils
This repository includes utilities to build and run CUDA Docker images.\
See ```samples/``` for a list of Dockerfile examples.

### Cudock
Generate or build CUDA Dockerfiles given CUDA version(s).

```sh
$ usage: ./cudock (generate|build) [version...]
```
```sh
$ ./cudock generate 7.0 7.5
'cuda-7.0' successfully generated
'cuda-7.5' successfully generated

$ ls -R
.: cuda-7.0  cuda-7.5
./cuda-7.0: Dockerfile
./cuda-7.5: Dockerfile
```
```sh
$ ./cudock build 7.5

$ docker images
REPOSITORY             TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
cuda                   7.5                 5eba96294224        About an hour ago   2.146 GB
cuda                   latest              5eba96294224        About an hour ago   2.146 GB
```
Alternatively, one can build a CUDA image directly from this repository:
```
$ docker build -t "cuda:latest" -f "cuda-7.5" github.com/nvidia-docker
```

### nvidia-docker

NVIDIA CUDA docker wrapper

This script is analogous to ```docker```, except that it will take care of setting up the NVIDIA host driver environment within Docker containers.\
GPUs are exported through a list of comma-separated IDs using the environment variable ```GPU```.\
Note thas the numbering is similar to the one reported by ```nvidia-smi```.

```sh
$ GPU=0,1 ./nvidia-docker <docker-options>
```
