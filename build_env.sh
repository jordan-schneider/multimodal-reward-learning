#!/bin/bash
if [ ! -d $HOME/miniconda/envs/mrl ]
then
	conda env create -f environment.yml
	conda run -n mrl pip install -r requirements-problematic.txt
	conda run -n mrl pip install -e .
fi
