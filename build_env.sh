#!/bin/bash
if [ ! -d $HOME/miniconda/envs/mrl ]
then
	conda env create -f environment.yml
fi
conda run -n mrl pip install --upgrade -r requirements-problematic.txt
conda run -n mrl pip install -e .
