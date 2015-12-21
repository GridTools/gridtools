#!/usr/bin/env bash

#
# install needed Python packages
#
apt-get update
apt-get install -y cmake                \
                   libfreetype6-dev     \
                   libpng12-dev         \
                   pkg-config           \
                   python3              \
                   python3-dev          \
                   python-virtualenv    \
                   qt4-bin-dbg          \
                   qt-sdk
#
# create the virtual environment
#
cd /home/vagrant
virtualenv --python=/usr/bin/python3.4 venv
source venv/bin/activate
pip install --upgrade pip distribute

#
# install Gridtools dependencies
#
pip install -r /vagrant/python/requirements.txt

#
# set the backend for matplotlib to Qt4/PySide
#
cat ./venv/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc | sed -e 's/^backend.*: pyside/backend : Qt4Agg/g' | sed -e 's/^#backend.qt4.*/backend.qt4 : PySide/g' > /tmp/.matplotlibrc
cp /tmp/.matplotlibrc ./venv/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc

#
# change virtualenv ownership to regular user
#
deactivate
chown --recursive vagrant:vagrant /home/vagrant/venv

#
# environment variables setup for regular user
#
echo "#                                             " >> /home/vagrant/.bashrc
echo "# environment setup for Gridtools development " >> /home/vagrant/.bashrc
echo "#                                             " >> /home/vagrant/.bashrc
echo "export CXX=/usr/bin/g++                       " >> /home/vagrant/.bashrc
echo "export BOOST_ROOT=/usr/local                  " >> /home/vagrant/.bashrc
echo "export CUDATOOLKIT_HOME=/usr/local/cuda-7.0   " >> /home/vagrant/.bashrc
echo "export GRIDTOOLS_ROOT=/vagrant                " >> /home/vagrant/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib         " >> /home/vagrant/.bashrc
echo "export PATH=${PATH}:${CUDATOOLKIT_HOME}/bin   " >> /home/vagrant/.bashrc

#
# remove unused packages
#
apt-get remove --purge -y libfreetype6-dev  \
                          libpng12-dev      \
                          python3-dev       \
                          qt-sdk            \
                          xterm
apt-get autoremove -y

#
# install CUDA 7.0
#
wget -q -O cuda_7.0.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.run
./cuda_7.0.run --verbose --samples --samplespath=/home/vagrant --silent --toolkit
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
