#FROM python:3.4-slim
FROM ubuntu:14.04

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

# environment variables
ENV DEBIAN_FRONTEND noninteractive

# install software for SSH access
RUN apt-get update --fix-missing                 && \
    apt-get install -y openssh-server  \
                       sudo            \
                       xauth           \
                    --no-install-recommends

# install dependencies for GridTools
RUN apt-get install -y autotools-dev        \
                       build-essential      \
                       cmake                \
                       gcc-4.8              \
                       gfortran-4.8         \
                       libbz2-dev           \
                       libicu-dev           \
                       libfreetype6-dev     \
                       liblapack-dev        \
                       libxft-dev           \
                       libzip-dev           \
                       libzmq-dev           \
                       locales              \
                       python3-dev          \
                       python3-pip          \
                       qt-sdk               \
                       qt4-bin-dbg          \
                       wget                 \
                    --no-install-recommends     && \
    rm -rf /var/lib/apt/lists
RUN pip3 install virtualenv

# user and locale configuration
RUN useradd docker                                                  && \
    echo "ALL            ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime             && \
    dpkg-reconfigure locales

WORKDIR /home/docker
ENV     HOME /home/docker
ENV     LC_ALL C.UTF-8

# SSH configuration
ADD  ssh/id_rsa.pub .ssh/authorized_keys
RUN  chmod 700  $HOME/.ssh                       && \
     chmod 600  $HOME/.ssh/authorized_keys       && \
     mkdir      /var/run/sshd                    && \
     chmod 0755 /var/run/sshd

# change ownership to non-root user
RUN chown --recursive docker:docker $HOME

# configure CUDA
USER docker
RUN  echo "CUDATOOLKIT_HOME=/usr/local/cuda" >> $HOME/.profile

# include GridTools source
ADD gridtools_ccd7860 gridtools
RUN echo "GRIDTOOLS_ROOT=$HOME/gridtools" >> $HOME/.profile

# install gridtools4py requirements inside a virtual environment
RUN virtualenv $HOME/venv                                                   && \
    . $HOME/venv/bin/activate                                               && \
    cd $HOME                                                                && \
    pip3 install --upgrade wheel                                            && \
    wget -q -O PySide-1.2.2.tar.gz https://pypi.python.org/packages/source/P/PySide/PySide-1.2.2.tar.gz && \
    tar -xvzf PySide-1.2.2.tar.gz                                           && \
    cd PySide-1.2.2                                                         && \
    python3 setup.py bdist_wheel --qmake=$( which qmake ) --jobs=4          && \
    pip3 install ./dist/PySide-1.2.2-cp34-cp34m-linux_x86_64.whl            && \
    python3 $HOME/venv/bin/pyside_postinstall.py -install                   && \
    rm -rf $HOME/PySide-1.2.2.tar.gz $HOME/PySide-1.2.2                     && \
    cd $HOME/gridtools/python                                               && \
    pip3 install -r requirements.txt                                        && \
    pip3 install --upgrade pyzmq
ADD matplotlibrc $HOME/venv/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc

#
# the following commands should be run as root
#
USER root

# install CUDA 7.0
RUN wget -q -O - http://developer.download.nvidia.com/compute/cuda/repos/GPGKEY | apt-key add - && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update --fix-missing

ENV CUDA_VERSION 7.0
LABEL com.nvidia.cuda.version="7.0"

RUN apt-get install -y --no-install-recommends --force-yes "cuda-toolkit-7.0" gpu-deployment-kit

RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# install Boost Libs
ENV BOOST_ROOT /usr/local
RUN echo "BOOST_ROOT=/usr/local" >> $HOME/.profile
RUN wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?use_mirror=kent' && \
    tar xzvf boost_1_58_0.tar.gz                                    && \
    cd boost_1_58_0                                                 && \
    ./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local        && \
    ./b2 -j4 cxxflags="-std=c++11" install                          && \
    echo "$BOOST_ROOT/lib" >> /etc/ld.so.conf.d/boost-1.58.0.conf   && \
    ldconfig -v                                                     && \
    cd ..                                                           && \
    rm -rf boost_1_58_0.tar.gz boost_1_58_0

# remove unused packages
RUN apt-get remove --purge -y libfreetype6-dev  \
                              libpng12-dev      \
                              libxft-dev        \
                              libzip-dev        \
                              libzmq-dev        \
                              python3-dev       \
                              qt-sdk            \
                              xterm
RUN apt-get autoremove -y

# change ownership to non-root user
RUN chown --recursive docker:docker $HOME

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
