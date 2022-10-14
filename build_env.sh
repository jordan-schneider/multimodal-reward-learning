#!/bin/bash
if ! mamba env list | grep -q "mrl" 
then
	mamba env create -f environment.yml
fi
mamba run -n mrl pip install --force-reinstall git+https://github.com/jordan-schneider/linear-procgen.git
mamba run -n mrl pip install -e .
