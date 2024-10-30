#!/bin/sh
set -x #echo on
set -e #exit on error
export CUDA_VISIBLE_DEVICES=-1
# clear files
pacemaker -c
# initial fit
pacemaker
# upfit
pacemaker -p output_potential.yaml
# active set
pace_activeset output_potential.yaml -d fitting_data_info.pckl.gzip
# data augmentation
pace_augment output_potential.yaml -d fitting_data_info.pckl.gzip -a output_potential.asi  -mss 5
# upfit
pacemaker input_aug.yaml -p output_potential.yaml
# active set
pace_activeset output_potential.yaml -d fitting_data_info.pckl.gzip
# auto core-rep
pace_corerep output_potential.yaml -a output_potential.asi -d fitting_data_info.pckl.gzip
# utilities
pace_yaml2yace output_potential.yaml
pace_info output_potential.yaml
pace_timing output_potential.yaml
