import os
from typing import Optional


def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))


def get_artifact_path(file_name: Optional[str] = None):
    path = os.path.join(get_base_path(), "artifacts")
    if file_name is not None:
        path = os.path.join(path, file_name)
    return path


def get_refinement_indices_path(file_name: Optional[str] = None):
    path = os.path.join(get_base_path(), "refinement_indices")
    if file_name is not None:
        path = os.path.join(path, file_name)
    return path
