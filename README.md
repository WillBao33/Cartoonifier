# Fine Art Portrait
This is the official github repo of Fine Art Portrait.

Fine Art Portrait is an application that turns a photographic portrait into a fine art portrait:

<img src=./cartoon.png width="100%" height="100%">

## Platform information
Hardware info 
```
GPU: NVIDIA RTX 3080Ti
CPU: AMD Ryzen 9 5950X
RAM: 32G
SSD: 512G + 2T
```

Software info
```
Ubuntu 20.04
Python 3.8
CUDA 11.6
Docker 20.10.18
NVIDIA-driver 510.85.02
OpenCV 4.5.5
matplotlib 3.1.2
numpy 1.23.0
```
Make sure you have Docker installed.

## Instruction 
```
git clone https://github.com/WillBao33/Cartoonifier.git
```
### Run with Docker
Building the docker image would take a while due to installation of OpenCV, so feel free to grab a cup of coffee :)

To build the Docker image and execute the application, run:
```
cd ./Fine-Art-Portrait
docker build -t portrait .
docker run --rm -it    --user=$(id -u)    --env="DISPLAY"    --workdir=/opt/build    --volume="$PWD":/app    --volume="/etc/group:/etc/group:ro"    --volume="/etc/passwd:/etc/passwd:ro"    --volume="/etc/shadow:/etc/shadow:ro"    --volume="/etc/sudoers.d:/etc/sudoers.d:ro"    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" portrait:latest bash
cd Fine-Art-Portrait
python3 portrait.py -i ./test_1.jpg
```
### Run without Docker
Make sure you have the required dependencies: OpenCV, matplotlib, and numpy.

You can install OpenCV following this [link](https://vitux.com/opencv_ubuntu/)
```
cd ./Fine-Art-Portrait
pip3 install -r requirements.txt
python3 portrait.py -i ./test_1.jpg
```
