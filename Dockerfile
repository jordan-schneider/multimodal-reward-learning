# Build using the standard pytorch base image
FROM docker.io/library/python:3.8-slim
RUN apt update && apt install -y git default-jre cmake g++ qtbase5-dev mpich && rm -rf /var/lib/apt/lists/*
WORKDIR .
COPY requirements.txt requirements.txt
COPY requirements-problematic.txt requirements-problematic.txt

RUN python -m pip install mpi4py
RUN python -m pip install -r requirements.txt
RUN python -m pip install -r requirements-problematic.txt

# Copy over your python code
COPY mrl/ mrl/
COPY setup.py setup.py

RUN python -m pip install .

# The command that will actually run. Make sure you actually copy the file over.
ENTRYPOINT ["mpiexec", "-np", "1", "python3.8", "mrl/model_training/train_oracles.py", "--path=data/", "--env-name=miner", "--seed=7892365", "--replications=1", "--overwrite", "--total-interactions=2000", "--n-parallel-envs=1"]