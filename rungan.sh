#!/bin/bash

#PBS -V
#PBS -N MNIST_GAN
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l place=free
#PBS -q production


cd $PBS_O_WORKDIR
python ~/playing-with-gan/mnist_gan.py

