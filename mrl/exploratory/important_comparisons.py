# %%
import os
import pickle as pkl
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

os.chdir("/home/joschnei/multimodal-reward-learning")


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size, dtype=p.dtype)
    return s


# %%
ROOTDIR: Final = Path("data/near-original-reward/7/prefs/compare-new")
N_TRIALS: Final = 10
dispersion_per_trial = [
    pkl.load((ROOTDIR / f"trial-{i}/dispersion_gt.pkl").open("rb"))
    for i in range(N_TRIALS)
]
modalities = dispersion_per_trial[0].keys()
orders_per_trial = [
    {
        modality: np.load((ROOTDIR / f"trial-{i}/{modality}_order.npy").open("rb"))
        for modality in modalities
    }
    for i in range(N_TRIALS)
]
# %%
drop_in_dispersion = [
    {
        modality: np.concatenate(([0], dispersion[:-1] - dispersion[1:]))
        for modality, dispersion in per_modality.items()
    }
    for per_modality in dispersion_per_trial
]
drop_in_dispersion = [
    {
        modality: drop[invert_permutation(orders[modality])]
        for modality, drop in drops.items()
    }
    for orders, drops in zip(orders_per_trial, drop_in_dispersion)
]

# %%
for modality in modalities:
    d = np.stack([drop[modality] for drop in drop_in_dispersion])
    df = pd.DataFrame(d).melt(var_name="comparison", value_name="drop")
    df = df.set_index("comparison")

    comparisons = df.groupby("comparison")
    mean_drop = comparisons.mean()
    good_comparisons = mean_drop >= mean_drop.quantile(0.99)
    big_df = df.loc[good_comparisons["drop"]].reset_index()

    sns.catplot(data=big_df, x="comparison", y="drop", kind="box")
    plt.title(f"10 biggest dispersion drops from {modality} comparisons")
    plt.show()
    plt.close()

# -> Conclusion is basically that there aren't particularly important comparisons independent of
# order, so our variance across orders should be smallish, which seems plasuible given 10 trial
# results
