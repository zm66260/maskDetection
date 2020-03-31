# Face Mask Detection

Since the Covid-19 pandemic has become a serious gloab health crisis, this project aims to encourge peolple wearing mask to protect themseleves and others.

# Introduction
The engine of this project is a well trained pytorch model provided by 
https://github.com/AIZOOTech/FaceMaskDetection. The raw project is entirely python-based. In this project, I converted the pytorch model to C++ version and rewritted inference code in C++ to implement real-time face tracking with camera.

# Requirements
libtorch1.1
openc3.4.5
xtensor

# Installation
git clone https://github.com/zm66260/maskDetection
cd maskDetection
mkdir build && cd build
cmake ..
make

# TODO
Implement other face detection model to further improve the accuracy.

