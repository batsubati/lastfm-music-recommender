import zipfile

import pytest

from src.data import _extract_archive_safely


def test_archive_path_traversal_is_rejected(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create an archive member that tries to write outside data_dir.
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "should not be written")

    # The unsafe archive must be rejected before extraction.
    with pytest.raises(RuntimeError, match="Unsafe archive path rejected"):
        _extract_archive_safely(archive_path, data_dir)

    assert not (tmp_path / "outside.txt").exists()
