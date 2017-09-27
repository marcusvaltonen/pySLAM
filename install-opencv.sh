#!/bin/bash
# Follows guide on http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

function fetch_pip() {
cd ~
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
}

# Check Ubuntu 16.04.1 LTS
[ '16.04' == `lsb_release -s -r` ] || { echo >&2 "This only works for Ubuntu 16.04."; exit 1; }

# Check shell
rc_file="~/.`echo $SHELL | sed -e 's/\/.*\///g'`rc"
echo "This will make changes in $rc_file"

# Update system
sudo apt-get update && sudo apt-get upgrade

# Install dependencies
sudo apt-get install -y build-essential cmake pkg-config libjpeg8-dev \
                        libtiff5-dev libjasper-dev libpng12-dev \
                        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                        libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev \
                        gfortran python2.7-dev python3.5-dev \

# Download OpenCV including contrig (non-free)
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip

# Check if pip is installed or else install it
command -v pip >/dev/null 2>&1 || { echo >&2 "Could not find pip. Will fetch it for you."; fetch_pip; }

sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/get-pip.py ~/.cache/pip

# Edit rc file
echo -e "\n# virtualenv and virtualenvwrapper" >> $rc_file
echo "export WORKON_HOME=$HOME/.virtualenvs" >> $rc_file
echo "source /usr/local/bin/virtualenvwrapper.sh" >> $rc_file

# Install venv
mkvirtualenv cv -p python2
workon cv

# Install numpy
pip install numpy

# Build OpenCV
cd ~/opencv-3.1.0/
mkdir build
cd build

# NB: There is a matlab flag that you can turn off
# If it does not find stdlib.h try
# -DENABLE_PRECOMPILED_HEADERS=OFF
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
    -D BUILD_EXAMPLES=ON \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D WITH_MATLAB=OFF ..

# After this
# make -j4
# sudo make install
# sudo ldconfig
# ln -s /usr/local/lib/python2.7/site-packages/cv2.so ~/.virtualenvs/cv/lib/python2.7/site-packages/cv2.so

pip install pillow
pip install matplotlib
pip install scipy
