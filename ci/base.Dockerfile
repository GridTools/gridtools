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


RUN wget --quiet https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz && \
    echo be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b boost_1_85_0.tar.gz > boost_hash.txt && \
    sha256sum -c boost_hash.txt && \
    tar xzf boost_1_85_0.tar.gz && \
    mv boost_1_85_0/boost /usr/local/include/ && \
    rm boost_1_85_0.tar.gz boost_hash.txt

ENV BOOST_ROOT /usr/local/
ENV CUDA_HOME /usr/local/cuda
ENV CUDA_ARCH=${CUDA_ARCH}
