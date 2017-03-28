#!/usr/bin/env python

#PBS -V
#PBS -l select=1:ncpus=1:ngpus=1

import keras
import keras.backend
print(keras.backend.backend())
