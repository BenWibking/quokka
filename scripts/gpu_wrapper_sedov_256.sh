#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_256.in
