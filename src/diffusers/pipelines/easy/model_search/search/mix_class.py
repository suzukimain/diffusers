from .pipeline_search_for_HuggingFace import HFSearchPipeline
from .pipeline_search_for_Civitai import CivitaiSearchPipeline

class Config_Mix(
    HFSearchPipeline,
    CivitaiSearchPipeline
    ):
    # Resolve multiple inheritance order (MRO) error
    def __init__(self):
        super().__init__()
