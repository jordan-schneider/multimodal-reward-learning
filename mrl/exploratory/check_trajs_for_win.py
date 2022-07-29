#%%
import gc
import pickle as pkl
from pathlib import Path
from typing import Generator, Sequence

from mrl.dataset.trajectory_db import FeatureDataset


#%%
def find_trajs(datadir: Path) -> Generator[FeatureDataset, None, None]:
    i = 0
    path = datadir / f"trajectories_{i}.pkl"
    while path.exists():
        yield pkl.load(path.open("rb"))
        i += 1


#%%
def count_finishes(datadirs: Sequence[Path]) -> int:
    count = 0
    for datadir in datadirs:
        for dataset in find_trajs(datadir):
            gc.collect()
            count += dataset.df["total_feature"].apply(lambda x: x[5]).sum()
    return count


#%%
rootdir = Path("/nfs/homes/joschnei/multimodal-reward-learning/data/miner")
datadirs = [rootdir / f"{i}" for i in range(1, 6)]


#%%
count_finishes(datadirs)
gc.collect()
