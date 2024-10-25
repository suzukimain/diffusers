from .pipeline_search_for_HuggingFace import HFSearchPipeline
from .pipeline_search_for_Civitai import CivitaiSearchPipeline
from .pipeline_search_for_hub import ModelSearchPipeline

class Config_Mix(
    HFSearchPipeline,
    CivitaiSearchPipeline,
    ModelSearchPipeline
    ):
    # Resolve multiple inheritance order (MRO) error
    def __init__(self):
        super().__init__()