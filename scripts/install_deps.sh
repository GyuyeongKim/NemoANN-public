#!/bin/bash
set -e

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake git curl python3-pip \
  libaio-dev libgoogle-perftools-dev libmkl-full-dev libeigen3-dev libboost-program-options-dev
pip3 install --user numpy faiss-cpu
