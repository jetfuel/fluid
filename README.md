# PaddlePaddle Fluid and Its Operator Base

[![Build Status](https://travis-ci.org/wangkuiyi/fluid.svg?branch=develop)](https://travis-ci.org/wangkuiyi/fluid)

This repo contains the ongoing progress of extracting 6PaddlePaddle Fluid, a subsystem of [Paddle](https://github.com/PaddlePaddle/Paddle), out from Paddle.

After the work, it will be migrated to github.com/PaddlePaddle/fluid.

## Motivations

PaddlePaddle Fluid and other PaddlePaddle subsystems live in the Github repo [Paddle](https://github.com/PaddlePaddle/Paddle).  When we are developing the new subsystem [Tape](https://github.com/PaddlePaddle/tape), we'd thought that we can simply make Tape rely on Paddle so could it reuse Fluid's operator base.  However, we encounter some difficulties:

1. Tape has to be cross-platform -- from Raspberry Pi to NVIDIA PX2/3, but Paddle relies on Linux and particularly Docker.  Fluid should be able to build without Docker, so we have to extract it out from the rest part of Paddle.
1. Another benefit of extracting Fluid out from Paddle repo is faster continuous integration (CI). For some reasons, it takes a few minutes for the CI of this repo to run, but half or close to an hour for the CI of Paddle.
1. We'd remove some unnecessary third-party dependencies from Fluid source code to make it cross-platform.
1. However, Fluid's source code contains many cyclic dependencies that we'd have to untangle.
1. The CMake-based build system of Paddle contains out-of-control legacy code and we'd like to switch to [bazel.cmake](https://github.com/gangliao/bazel.cmake/).

For above and some more reasons, we decided to clean up Fluid in a separate repo, which is here.

## How to Build

```bash
cd ~/work
git clone --recursive https://github.com/wangkuiyi/fluid
cd fluid
mkdir build
cd build
cmake -DWITH_GPU=OFF .. # or cmake .. if you have a GPU
make -j10 -k
ctest
```
