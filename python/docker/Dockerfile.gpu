FROM cuda:latest

MAINTAINER Lucas Benedicic <benedicic@cscs.ch>

# environment variables
ENV DEBIAN_FRONTEND noninteractive

# install software for SSH access
RUN apt-get update                              && \
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
                       python3-dev          \
                       python3-pip          \
                       qt-sdk               \
                       qt4-bin-dbg          \
                       wget                 \
                    --no-install-recommends     && \
    rm -rf /var/lib/apt/lists

# user and locale configuration
RUN useradd docker                                                  && \
    echo "ALL            ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime             && \
    dpkg-reconfigure locales                                        && \
    locale-gen en_US.UTF-8                                          && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

WORKDIR /home/docker
ENV     HOME /home/docker
ENV     LC_ALL en_US.UTF-8

# SSH configuration
ADD ssh/id_rsa.pub .ssh/authorized_keys
RUN chmod 700  $HOME/.ssh                       && \
    chmod 600  $HOME/.ssh/authorized_keys       && \
    mkdir      /var/run/sshd                    && \
    chmod 0755 /var/run/sshd

# install Boost Libs
ENV BOOST_ROOT /usr/local
RUN echo "BOOST_ROOT=/usr/local" >> $HOME/.profile
RUN wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?use_mirror=kent' && \
    tar xzvf boost_1_58_0.tar.gz                                                        && \
    cd boost_1_58_0                                                                     && \
    ./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local        && \
    ./b2 -j4 cxxflags="-std=c++11" install                                              && \
    echo "$BOOST_ROOT/lib" >> /etc/ld.so.conf.d/boost-1.58.0.conf                       && \
    ldconfig -v                                                                         && \
    cd ..                                                                               && \
    rm -rf boost_1_58_0.tar.gz boost_1_58_0

# configure CUDA
RUN echo "CUDATOOLKIT_HOME=/usr/local/cuda" >> $HOME/.profile

# include GridTools source
ADD gridtools_ccd7860 gridtools
RUN echo "GRIDTOOLS_ROOT=$HOME/gridtools" >> $HOME/.profile

# install gridtools4py requirements
# -- pyside --
RUN cd $HOME                                                        && \
    pip3 install wheel                                              && \
    wget -q -O PySide-1.2.2.tar.gz https://pypi.python.org/packages/source/P/PySide/PySide-1.2.2.tar.gz && \
    tar -xvzf PySide-1.2.2.tar.gz                                   && \
    cd PySide-1.2.2                                                 && \
    python3 setup.py bdist_wheel --qmake=$( which qmake ) --jobs=4  && \
    pip3 install ./dist/PySide-1.2.2-cp34-cp34m-linux_x86_64.whl    && \
    python3 /usr/local/bin/pyside_postinstall.py -install           && \
    rm -rf $HOME/PySide-1.2.2.tar.gz $HOME/PySide-1.2.2

# -- and the rest ---
RUN cd $HOME/gridtools/python                                       && \
    pip3 install -r requirements.txt
ADD matplotlibrc /usr/local/lib/python3.4/dist-packages/matplotlib/mpl-data/matplotlibrc

# remove unused packages
apt-get remove --purge -y libfreetype6-dev  \
                          liblapack-dev     \
                          libpng12-dev      \
                          libxft-dev        \
                          libzip-dev        \
                          libzmq-dev        \
                          python3-dev       \
                          qt-sdk            \
                          xterm
apt-get autoremove -y

# change ownership to non-root user
RUN chown --recursive docker:docker $HOME

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
