from ..loaders.single_file_utils import infer_diffusers_model_type,load_single_file_checkpoint

AUTO_TEXT2IMAGE_SINGLE_FILE_CHECKPOINT_MAPPING = {
    "xl_base": 1024,
    "xl_refiner": 1024,
    "xl_inpaint": None,
    "playground-v2-5": DiffusionPipeline,
    "upscale": 512,
    "inpainting": 512,
    "inpainting_v2": 512,
    "controlnet": 512,
    "v2": StableDiffusionPipeline,
    "v1": StableDiffusionPipeline,
}
AUTO_IMAGE2IMAGE_SINGLE_FILE_CHECKPOINT_MAPPING = {
    "xl_base": 1024,
    "xl_refiner": 1024,
    "xl_inpaint": 1024,
    "playground-v2-5": 1024,
    "upscale": 512,
    "inpainting": 512,
    "inpainting_v2": 512,
    "controlnet": 512,
    "v2": 768,
    "v1": 512,
}
AUTO_INPAINT_SINGLE_FILE_CHECKPOINT_MAPPING = {
    "xl_base": 1024,
    "xl_refiner": 1024,
    "xl_inpaint": 1024,
    "playground-v2-5": 1024,
    "upscale": 512,
    "inpainting": 512,
    "inpainting_v2": 512,
    "controlnet": 512,
    "v2": StableDiffusionInpaintPipeline,
    "v1": StableDiffusionInpaintPipeline,
}

import importlib
import inspect
import os

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, validate_hf_hub_args
from packaging import version

from ..utils import deprecate, is_transformers_available, logging
from ..loaders.single_file_utils import (
    SingleFileComponentError,
    _is_legacy_scheduler_kwargs,
    _is_model_weights_in_cached_folder,
    _legacy_load_clip_tokenizer,
    _legacy_load_safety_checker,
    _legacy_load_scheduler,
    create_diffusers_clip_model_from_ldm,
    create_diffusers_t5_model_from_checkpoint,
    fetch_diffusers_config,
    fetch_original_config,
    is_clip_model_in_single_file,
    is_t5_in_single_file,
    load_single_file_checkpoint,
)
from ..loaders.single_file import _download_diffusers_model_config_from_hub,_infer_pipeline_config_dict,SINGLE_FILE_OPTIONAL_COMPONENTS,load_single_file_sub_model


from ..configuration_utils import ConfigMixin
from ..utils import is_sentencepiece_available
from .aura_flow import AuraFlowPipeline
from .cogview3 import CogView3PlusPipeline
from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from .flux import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
)
from .hunyuandit import HunyuanDiTPipeline
from .kandinsky import (
    KandinskyCombinedPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintCombinedPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
)
from .kandinsky2_2 import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintCombinedPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
)
from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
from .lumina import LuminaText2ImgPipeline
from .pag import (
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
from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from .stable_diffusion_3 import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
)
from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline
from .pipeline_utils import DiffusionPipeline


from ..utils import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name






SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = {
    "xl_base": StableDiffusionXLPipeline,
    "xl_refiner": StableDiffusionXLPipeline,
    "xl_inpaint": None,
    "playground-v2-5": StableDiffusionXLPipeline,
    "upscale": None,
    "inpainting": None,
    "inpainting_v2": None,
    "controlnet": StableDiffusionControlNetPipeline,
    "v2": StableDiffusionPipeline,
    "v1": StableDiffusionPipeline,
}
SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = {
    "xl_base": StableDiffusionXLImg2ImgPipeline,
    "xl_refiner": StableDiffusionXLImg2ImgPipeline,
    "xl_inpaint": None,
    "playground-v2-5": StableDiffusionXLImg2ImgPipeline,
    "upscale": None,
    "inpainting": None,
    "inpainting_v2": None,
    "controlnet": StableDiffusionControlNetImg2ImgPipeline,
    "v2": StableDiffusionImg2ImgPipeline,
    "v1": StableDiffusionImg2ImgPipeline,

}

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = {
    "xl_base": None,
    "xl_refiner": None,
    "xl_inpaint": StableDiffusionXLInpaintPipeline,
    "playground-v2-5": None,
    "upscale": None,
    "inpainting": StableDiffusionInpaintPipeline,
    "inpainting_v2": StableDiffusionInpaintPipeline,
    "controlnet": StableDiffusionControlNetInpaintPipeline,
    "v2": None,
    "v1": None,
}

INPAINT_PIPELINE_KEYS = [
    "xl_inpaint",
    "inpainting",
    "inpainting_v2",
    ]



def _load_single_file_checkpoint(pretrained_model_link_or_path,pipeline_mapping,**kwargs):
    r"""
    Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
    format. The pipeline is set in evaluation mode (`model.eval()`) by default.

    Parameters:
        pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:
                - A link to the `.ckpt` file (for example
                  `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                - A path to a *file* containing all pipeline weights.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model with another dtype.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.

        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        original_config_file (`str`, *optional*):
            The path to the original config file that was used to train the model. If not provided, the config file
            will be inferred from the checkpoint file.
        config (`str`, *optional*):
            Can be either:
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                  component configs in Diffusers format.
        checkpoint (`dict`, *optional*):
            The loaded state dictionary of the model.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
    """
    checkpoint = load_single_file_checkpoint(pretrained_model_link_or_path)
    model_type = infer_diffusers_model_type(checkpoint)
    pipeline_class = pipeline_mapping[model_type]
    if pipeline_class is None:
        if pipeline_class in INPAINT_PIPELINE_KEYS:
            raise ValueError("`inpaint` is only supported in `AutoPipelineForInpainting.from_pretrained`")
        else:
            raise ValueError(f"{pipeline_class} is not supported.")
    else:
        return pipeline_class.from_single_file(pretrained_model_link_or_path, checkpoint=checkpoint, **kwargs)






from collections import OrderedDict

SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", StableDiffusionXLPipeline),
        ("xl_refiner", StableDiffusionXLPipeline),
        ("xl_inpaint", None),
        ("playground-v2-5", StableDiffusionXLPipeline),
        ("upscale", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("controlnet", StableDiffusionControlNetPipeline),
        ("v2", StableDiffusionPipeline),
        ("v1", StableDiffusionPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", StableDiffusionXLImg2ImgPipeline),
        ("xl_refiner", StableDiffusionXLImg2ImgPipeline),
        ("xl_inpaint", None),
        ("playground-v2-5", StableDiffusionXLImg2ImgPipeline),
        ("upscale", None),
        ("inpainting", None),
        ("inpainting_v2", None),
        ("controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("v2", StableDiffusionImg2ImgPipeline),
        ("v1", StableDiffusionImg2ImgPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = OrderedDict(
    [
        ("xl_base", None),
        ("xl_refiner", None),
        ("xl_inpaint", StableDiffusionXLInpaintPipeline),
        ("playground-v2-5", None),
        ("upscale", None),
        ("inpainting", StableDiffusionInpaintPipeline),
        ("inpainting_v2", StableDiffusionInpaintPipeline),
        ("controlnet", StableDiffusionControlNetInpaintPipeline),
        ("v2", None),
        ("v1", None),
    ]
)

INPAINT_PIPELINE_KEYS = [
    "xl_inpaint",
    "inpainting",
    "inpainting_v2",
]

def _load_single_file_checkpoint(pretrained_model_link_or_path, pipeline_mapping, **kwargs):
    r"""
    Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
    format. The pipeline is set in evaluation mode (`model.eval()`) by default.

    Parameters:
        pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:
                - A link to the `.ckpt` file (for example
                  `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                - A path to a *file* containing all pipeline weights.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model with another dtype.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        original_config_file (`str`, *optional*):
            The path to the original config file that was used to train the model. If not provided, the config file
            will be inferred from the checkpoint file.
        config (`str`, *optional*):
            Can be either:
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                  component configs in Diffusers format.
        checkpoint (`dict`, *optional*):
            The loaded state dictionary of the model.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
    """
    checkpoint = load_single_file_checkpoint(pretrained_model_link_or_path)
    model_type = infer_diffusers_model_type(checkpoint)
    pipeline_class = pipeline_mapping[model_type]
    if pipeline_class is None:
        if model_type in INPAINT_PIPELINE_KEYS:
            raise ValueError("`inpaint` is only supported in `AutoPipelineForInpainting.from_pretrained`")
        else:
            raise ValueError(f"{model_type} is not supported.")
    else:
        return pipeline_class.from_single_file(pretrained_model_link_or_path, checkpoint=checkpoint, **kwargs)
