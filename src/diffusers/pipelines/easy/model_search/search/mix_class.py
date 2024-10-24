from .search_hugface import HFSearch
from .search_civitai import CivitaiSearch
from ..search_utils.base_config import Basic_config


class Config_Mix(
    HFSearch,
    CivitaiSearch,
    Basic_config
    ):
    #fix MMO error
    def __init__(self):
        super().__init__()
