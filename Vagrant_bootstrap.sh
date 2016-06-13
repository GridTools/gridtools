#!/bin/bash

HOME_DIR=/home/vagrant
VENV_DIR=${HOME_DIR}/venv

#
# install the required dependencies
#
apt-get update
apt-get install -y chromium             \
                   cmake                \
                   libfreetype6-dev     \
                   libpng12-dev         \
                   pkg-config           \
                   python3              \
                   python3-dev          \
                   qt4-bin-dbg          \
                   qt-sdk               \
                   virtualenv           \
                --no-install-recommends

#
# create the virtual environment
#
virtualenv --python=/usr/bin/python3.4 ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install --upgrade pip distribute

#
# install Gridtools dependencies
#
pip --verbose install -r /vagrant/python/requirements.txt

#
# set the backend for matplotlib to Qt4/PySide
#
cat ${VENV_DIR}/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc | sed -e 's/^backend.*: pyside/backend : Qt4Agg/g' | sed -e 's/^#backend.qt4.*/backend.qt4 : PySide/g' > /tmp/.matplotlibrc
cp /tmp/.matplotlibrc ${VENV_DIR}/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc

#
# change virtualenv ownership to regular user
#
deactivate
chown --recursive vagrant:vagrant ${VENV_DIR}

#
# install CUDA 7.0
#
wget -q -O cuda_7.0.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.run
./cuda_7.0.run --verbose --samples --samplespath=${HOME_DIR} --silent --toolkit
rm cuda_7.0.run /tmp/cuda?install*

#
# install Boost 1.58
#
wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.58.0%2F&ts=1446134333&use_mirror=kent'
tar xvzf boost_1_58_0.tar.gz
cd boost_1_58_0
./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local
./b2
./b2 install
cd ..
rm -rf boost_1_58_0 boost_1_58_0.tar.gz

#
# environment setup for regular user
#
echo "#                                             " >> ${HOME_DIR}/.bashrc
echo "# environment setup for Gridtools development " >> ${HOME_DIR}/.bashrc
echo "#                                             " >> ${HOME_DIR}/.bashrc
echo "export CXX=/usr/bin/g++                       " >> ${HOME_DIR}/.bashrc
echo "export BOOST_ROOT=/usr/local                  " >> ${HOME_DIR}/.bashrc
echo "export CUDATOOLKIT_HOME=/usr/local/cuda-7.0   " >> ${HOME_DIR}/.bashrc
echo "export GRIDTOOLS_ROOT=/vagrant                " >> ${HOME_DIR}/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib         " >> ${HOME_DIR}/.bashrc
echo "export PATH=${PATH}:${CUDATOOLKIT_HOME}/bin   " >> ${HOME_DIR}/.bashrc
