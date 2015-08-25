FROM ubuntu:14.04

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

# environment variables
ENV DEBIAN_FRONTEND noninteractive

# install software for SSH and GridTools
RUN apt-get update                              && \
    apt-get install -y openssh-server  \
                       sudo            \
                       xauth           \
                    --no-install-recommends     && \
    apt-get install -y autotools-dev        \
                       build-essential      \
                       cmake                \
                       gcc-4.8              \
                       gfortran-4.8         \
                       git                  \
                       libbz2-dev           \
                       libzip-dev           \
                       libicu-dev           \
                       libfreetype6-dev     \
                       liblapack-dev        \
                       libopenblas-dev      \
                       libxft-dev           \
                       libzmq-dev           \
                       python3-dev          \
                       python3-pip          \
                       qt4-dev-tools        \
                       wget                     && \
    rm -rf /var/lib/apt/lists

# user and locale configuration
RUN useradd dev                                                     && \
    echo "ALL            ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime             && \
    dpkg-reconfigure locales                                        && \
    locale-gen en_US.UTF-8                                          && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

WORKDIR /home/dev
ENV     HOME /home/dev
ENV     LC_ALL en_US.UTF-8

# SSH configuration
ADD ssh/id_rsa.pub .ssh/authorized_keys
RUN chmod 700  $HOME/.ssh                       && \
    chmod 600  $HOME/.ssh/authorized_keys       && \
    mkdir      /var/run/sshd                    && \
    chmod 0755 /var/run/sshd

# install Boost Libs
ENV BOOST_ROOT /usr/local/boost-1.56.0
RUN wget -O boost_1_56_0.tar.gz http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.gz/download && \
    tar xzvf boost_1_56_0.tar.gz                                                    && \
    cd boost_1_56_0                                                                 && \
    ./bootstrap.sh --prefix=$BOOST_ROOT                                             && \
    NCORES=$( cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}' )     && \
    ./b2 --with=all -j $NCORES cxxflags="-std=c++11" --target=shared,static install && \
    echo "$BOOST_ROOT/lib" >> /etc/ld.so.conf.d/boost-1.56.0.conf                   && \
    ldconfig -v                                                                     && \
    cd ..                                                                           && \
    rm -rf boost_1_56_0.tar.gz boost_1_56_0

# install Nvidia driver
ENV CUDATOOLKIT_HOME /usr/local/cuda-7.0
RUN mkdir -p $HOME/cuda                                 && \
    cd $HOME/cuda                                       && \
    wget -O NVIDIA-Linux.run http://us.download.nvidia.com/XFree86/Linux-x86_64/352.21/NVIDIA-Linux-x86_64-352.21.run && \
    chmod +x NVIDIA-Linux.run                           && \
    ./NVIDIA-Linux.run --accept-license           \
                       --no-kernel-module         \
                       --no-kernel-module-source  \
                       --no-unified-memory        \
                       --no-install-vdpau-wrapper \
                       --silent                         && \
    rm NVIDIA-Linux.run

# install CUDA 7
RUN wget -O cuda_7.0.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
    chmod +x cuda_7.0.run                                                                                                  && \
    ./cuda_7.0.run --verbose --samples --samplespath=/home/dev/cuda --silent --toolkit --toolkitpath=$CUDATOOLKIT_HOME     && \
    rm cuda_7.0.run /tmp/cuda?install*

# include GridTools source
ADD gridtools_ccd7860 gridtools

# install GridTools requirements
# -- pyside --
RUN cd $HOME                                                        && \
    pip3 install wheel                                              && \
    wget -O PySide-1.2.2.tar.gz https://pypi.python.org/packages/source/P/PySide/PySide-1.2.2.tar.gz && \
    tar -xvzf PySide-1.2.2.tar.gz                                   && \
    cd PySide-1.2.2                                                 && \
    python3 setup.py bdist_wheel --qmake=$( which qmake ) --jobs=4  && \
    pip3 install ./dist/PySide-1.2.2-cp34-cp34m-linux_x86_64.whl    && \
    python3 /usr/local/bin/pyside_postinstall.py -install           && \
    rm -rf $HOME/PySide-1.2.2.tar.gz $HOME/PySide-1.2.2

# -- and the rest ---
RUN cd $HOME/gridtools/python                                      && \
    pip3 install -r requirements.txt
ADD matplotlibrc /usr/local/lib/python3.4/dist-packages/matplotlib/mpl-data/matplotlibrc

# change ownership to non-root user
RUN chown --recursive dev:dev $HOME

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
