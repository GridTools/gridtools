ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
    gfortran \
    g++ \
    gcc \
    strace \
    build-essential \
    tar \
    wget \
    curl \
    cmake \
    ca-certificates \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libreadline-dev \
    python3-dev \
    python3-pip \
    git \
    rustc \
    htop && \
    rm -rf /var/lib/apt/lists/*

ARG MPICH_VERSION=3.3.2
ARG MPICH_PATH=/usr/local
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure \
    --disable-fortran \
    --prefix=$MPICH_PATH && \
    make install -j32 && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}
RUN echo "${MPICH_PATH}/lib" >> /etc/ld.so.conf.d/cscs.conf && ldconfig

ENV CXX=${MPICH_PATH}/bin/mpicxx
ENV CC=${MPICH_PATH}/bin/mpicc

ENV CUDA_HOME /usr/local/cuda
ENV CUDA_ARCH=${CUDA_ARCH}
