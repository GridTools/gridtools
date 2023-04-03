
FROM gitpod/workspace-full

USER root

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${CLANG_VERSION} main" >> /etc/apt/sources.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${CLANG_VERSION} main" >> /etc/apt/sources.list \
    && wget -q -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-get update \
    && apt-get install -y libboost-all-dev ninja-build gfortran clang-15 libomp-15-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

ENV CXX=clang++-${CLANG_VERSION} CC=clang-${CLANG_VERSION}

ARG CMAKE_VERSION=3.26.2
RUN cd /tmp && \
    VNUM=$(echo ${CMAKE_VERSION} | awk -F \. {'print $1*1000+$2'}) && \
    LNAME=$([ ${VNUM} -gt 3019 ] && echo "linux" || echo "Linux") && \
    SRC="cmake-${CMAKE_VERSION}-${LNAME}-x86_64" && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${SRC}.tar.gz && \
    tar xzf ${SRC}.tar.gz && \
    cp -r ${SRC}/bin ${SRC}/share /usr/local/ && \
    rm -rf *

USER gitpod
