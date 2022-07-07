#!/bin/bash
if ! conda env list | grep -q "mrl" 
then
	conda env create -f environment.yml
fi
conda run -n mrl pip install --force-reinstall git+https://github.com/jordan-schneider/linear-procgen.git
conda run -n mrl pip install -e .
