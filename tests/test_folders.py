import shutil
from pathlib import Path

import pytest
from mrl.folders import HyperFolders


def test_check():
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )


def test_too_few_schema():
    with pytest.raises(RuntimeError) as err:
        folders = HyperFolders(
            rootdir=Path("tests/test_folder_root"), schema=["hyper_1"]
        )
    assert "no files" in str(err.value)


def test_too_many_schema():
    with pytest.raises(RuntimeError) as err:
        folders = HyperFolders(
            rootdir=Path("tests/test_folder_root"),
            schema=["hyper_1", "hyper_2", "hyper_3"],
        )
    assert "Found files" in str(err.value)


def test_update_schema_at_end():
    rootdir = Path("tests/test_folder_root")
    backup = Path("tests/backup")
    shutil.copytree(rootdir, backup)
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )
    folders.update_schema(["hyper_1", "hyper_2", "hyper_3"], [None, None, "hyper_3"])
    folders.check()
    assert Path(rootdir / "hyper_1" / "hyper_2" / "hyper_3" / "test.txt").is_file()
    shutil.rmtree(rootdir)
    backup.rename(rootdir)


def test_update_schema_at_beginning():
    rootdir = Path("tests/test_folder_root")
    backup = Path("tests/backup")
    shutil.copytree(rootdir, backup)
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )
    folders.update_schema(["hyper_3", "hyper_1", "hyper_2"], ["hyper_3", None, None])
    folders.check()
    assert Path(rootdir / "hyper_3" / "hyper_1" / "hyper_2" / "test.txt").is_file()
    shutil.rmtree(rootdir)
    backup.rename(rootdir)


def test_update_schema_in_middle():
    rootdir = Path("tests/test_folder_root")
    backup = Path("tests/backup")
    shutil.copytree(rootdir, backup)
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )
    folders.update_schema(["hyper_1", "hyper_3", "hyper_2"], [None, "hyper_3", None])
    folders.check()
    assert Path(rootdir / "hyper_1" / "hyper_3" / "hyper_2" / "test.txt").is_file()
    shutil.rmtree(rootdir)
    backup.rename(rootdir)


def test_update_schema_everywhere():
    rootdir = Path("tests/test_folder_root")
    backup = Path("tests/backup")
    shutil.copytree(rootdir, backup)
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )
    folders.update_schema(
        ["hyper_3", "hyper_1", "hyper_4", "hyper_2", "hyper_5"],
        ["hyper_3", None, "hyper_4", None, "hyper_5"],
    )
    folders.check()
    assert Path(
        rootdir / "hyper_3" / "hyper_1" / "hyper_4" / "hyper_2" / "hyper_5" / "test.txt"
    ).is_file()
    shutil.rmtree(rootdir)
    backup.rename(rootdir)


def test_update_schema_reorder():
    rootdir = Path("tests/test_folder_root")
    backup = Path("tests/backup")
    shutil.copytree(rootdir, backup)
    folders = HyperFolders(
        rootdir=Path("tests/test_folder_root"), schema=["hyper_1", "hyper_2"]
    )
    folders.update_schema(
        ["hyper_5", "hyper_4", "hyper_3", "hyper_2", "hyper_1"],
        ["hyper_5", "hyper_4", "hyper_3", None, None],
    )
    folders.check()
    assert Path(
        rootdir / "hyper_5" / "hyper_4" / "hyper_3" / "hyper_2" / "hyper_1" / "test.txt"
    ).is_file()
    shutil.rmtree(rootdir)
    backup.rename(rootdir)
