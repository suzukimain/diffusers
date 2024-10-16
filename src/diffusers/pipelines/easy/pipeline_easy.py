#main

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import diffusers




from ..pipeline_utils import DiffusionPipeline

from ...pipelines import (
    StableDiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    TextToVideoZeroPipeline,
    FlaxStableDiffusionPipeline,
    FlaxStableDiffusionImg2ImgPipeline,
    FlaxStableDiffusionInpaintPipeline,
    )


from ...configuration_utils import ConfigMixin
from ...utils import is_sentencepiece_available
from ..aura_flow import AuraFlowPipeline
from ..cogview3 import CogView3PlusPipeline
from ..controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)
from ..deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from ..flux import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
)
from ..hunyuandit import HunyuanDiTPipeline
from ..kandinsky import (
    KandinskyCombinedPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintCombinedPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
)
from ..kandinsky2_2 import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintCombinedPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
)
from ..kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
from ..latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
from ..lumina import LuminaText2ImgPipeline
from ..pag import (
    HunyuanDiTPAGPipeline,
    PixArtSigmaPAGPipeline,
    StableDiffusion3PAGPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    StableDiffusionControlNetPAGPipeline,
    StableDiffusionPAGImg2ImgPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    StableDiffusionXLControlNetPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusionXLPAGInpaintPipeline,
    StableDiffusionXLPAGPipeline,
)
from ..pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
from ..stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline
from ..stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from ..stable_diffusion_3 import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
)
from ..stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)


from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin





from model_path.perform_path_search import Search_cls
from model_path.mix_class import Basic_config


@dataclass
class AutoPipe_data:
    pipe_dict = {
        "torch":{
            "base" : StableDiffusionPipeline,
            "txt2img" : AutoPipelineForText2Image,
            "img2img" : AutoPipelineForImage2Image,
            "inpaint" : AutoPipelineForInpainting,
            "txt2video" : TextToVideoZeroPipeline,
        },
        "flax" : {
            "base" : FlaxStableDiffusionPipeline,
            "txt2img" : FlaxStableDiffusionPipeline,
            "img2img" : FlaxStableDiffusionImg2ImgPipeline,
            "inpaint" : FlaxStableDiffusionInpaintPipeline,
            "txt2video" : None,
        }

    }





class AutoPipeline(Search_cls,Basic_config):
    def __init__(
            self,
            search_word: str,
            pipe_type = "txt2img",
            auto: Optional[bool] = True,
            priority: Optional[str] = "hugface",
            branch: Optional[str] = "main",
            search_local_only: Optional[bool] = False,
            **keywords
            ):
        
        super().__init__()
        self.pipe_type = pipe_type
        self.priority = priority
        self.branch = branch
        self.search_local_only = search_local_only

        self.device = self.device_type_check()

    

    def pipeline_type(
            self,
            cls_or_name
            ):
        if isinstance(str, cls_or_name):
            if hasattr(diffusers, cls_or_name):
                return getattr(diffusers, cls_or_name)
            else:
                candidate = self.max_temper(cls_or_name,dir(diffusers))
                error_txt = f"Maybe {candidate}?" if candidate else ""
                raise ValueError(f"{cls_or_name} is not in diffusers.{error_txt}")
        
        elif hasattr(diffusers, cls_or_name.__name__):
            return cls_or_name
        
        else:
            candidate = self.max_temper(cls_or_name,dir(diffusers))
            error_txt = f"Maybe {candidate}?" if candidate else ""
            raise ValueError(f"{cls_or_name} is not in diffusers.{error_txt}")