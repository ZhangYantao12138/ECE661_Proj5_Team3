"""Portable build locations; preserve the current workspace without hardcoding a user."""
import os
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def workspace_root():
    explicit=os.environ.get('ECE661_WORKSPACE')
    if explicit:return Path(explicit).expanduser().resolve()
    if PROJECT_ROOT.parent.name=='recovered_gpu_v2' and PROJECT_ROOT.parent.parent.name=='work':
        return PROJECT_ROOT.parents[2]
    return PROJECT_ROOT


def delivery_dir():
    explicit=os.environ.get('ECE661_DELIVERY_DIR')
    return Path(explicit).expanduser().resolve() if explicit else workspace_root()/'outputs/ECE661_Final'
