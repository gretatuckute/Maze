#!/bin/sh
input_fname="beta-control-neural_stimset_D-S_light_truncated"
main_dir="/Users/gt/Documents/GitHub/Maze/maze_automate/"

cd $main_dir

/opt/anaconda3/envs/maze/bin/python distract.py $main_dir"/input_files/"$input_fname".txt" $main_dir"/output_files/"$input_fname".txt" -p params.txt --format ibex

