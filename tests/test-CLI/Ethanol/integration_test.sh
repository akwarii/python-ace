#!/bin/sh
set -x #echo on
set -e #exit on error
export CUDA_VISIBLE_DEVICES=-1
# clear files
../../../bin/pacemaker -c
# initial fit
../../../bin/pacemaker
# upfit
../../../bin/pacemaker -p output_potential.yaml
# active set
../../../bin/pace_activeset output_potential.yaml -d fitting_data_info.pckl.gzip
# data augmentation
../../../bin/pace_augment output_potential.yaml -d fitting_data_info.pckl.gzip -a output_potential.asi -mss 5 -nnmin 0.3
# upfit
../../../bin/pacemaker input_aug.yaml -p output_potential.yaml
# active set
../../../bin/pace_activeset output_potential.yaml -d fitting_data_info.pckl.gzip
# auto core-rep
../../../bin/pace_corerep output_potential.yaml -a output_potential.asi -d fitting_data_info.pckl.gzip
# utilities
../../../bin/pace_yaml2yace output_potential.yaml
../../../bin/pace_info output_potential.yaml
../../../bin/pace_timing output_potential.yaml
