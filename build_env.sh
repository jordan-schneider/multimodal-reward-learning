#!/bin/bash
conda env create -f environment.yml
conda run -n mrl pip install -r requirements-problematic.txt