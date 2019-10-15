#!/bin/bash

THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0',floatX=float32, device=gpu0, optimizer=fast_compile, exception_verbosity=high  python -m pdb train.py
