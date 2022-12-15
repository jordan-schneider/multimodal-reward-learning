from pathlib import Path

import numpy as np

from mrl.inference.results import Results


def test_getall():
    results = Results(Path("tests/test_results"))
    results.start("trial1")
    results.update("a", np.array([1, 2, 3]))
    results.update_dict("b", "state", np.array([4, 5, 6]))
    results.update_dict("b", "action", np.array([4, 5, 6]))
    results.update_dict("b", "reward", np.array([4, 5, 6]))
    results.update_dict("c", "state", np.arange(9).reshape(3, 3))
    results.update_dict("c", "action", np.arange(9).reshape(3, 3) + 9)
    results.update_dict("c", "reward", np.arange(9).reshape(3, 3) + 18)

    results.start("trial2")
    results.update("a", 3 + np.array([1, 2, 3]))
    results.update_dict("b", "state", 3 + np.array([4, 5, 6]))
    results.update_dict("b", "action", 3 + np.array([4, 5, 6]))
    results.update_dict("b", "reward", 3 + np.array([4, 5, 6]))
    results.update_dict("c", "state", 25 + np.arange(9).reshape(3, 3))
    results.update_dict("c", "action", 25 + np.arange(9).reshape(3, 3) + 9)
    results.update_dict("c", "reward", 25 + np.arange(9).reshape(3, 3) + 18)

    df_a = results.getall("a")
    print(df_a)
    assert np.array_equal(df_a.columns, ["trial", "time", "a"])
    assert len(df_a) == 6
    trial_1_a = df_a[df_a["trial"] == "trial1"]
    assert len(trial_1_a) == 3
    assert np.array_equal(trial_1_a["a"].values, [1, 2, 3])
    assert np.array_equal(trial_1_a["time"].values, [0, 1, 2])

    df_b = results.getall("b")
    print(df_b)
    assert np.array_equal(df_b.columns, ["trial", "time", "b", "modality"])
    assert len(df_b) == 18
    trial_1_b = df_b[df_b["trial"] == "trial1"]
    assert len(trial_1_b) == 9
    assert np.array_equal(trial_1_b["b"].values, [4, 5, 6, 4, 5, 6, 4, 5, 6])
    assert np.array_equal(trial_1_b["time"].values, [0, 1, 2, 0, 1, 2, 0, 1, 2])

    df_c = results.getall("c")
    print(df_c)
    assert np.array_equal(
        df_c.columns, ["trial", "time", "c_0", "c_1", "c_2", "modality"]
    )
    assert len(df_c) == 2 * 3 * 3
    trial_1_c = df_c[df_c["trial"] == "trial1"]
    assert len(trial_1_c) == 3 * 3
    assert np.array_equal(
        trial_1_c["c_0"].values, [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
    )
