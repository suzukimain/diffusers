from ..loaders.single_file_utils import infer_diffusers_model_type,load_single_file_checkpoint



from ..loaders.single_file_utils import (
    load_single_file_checkpoint,
)

from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
)
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)




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
        pipeline_mapping (`dict`):
            A mapping of model types to their corresponding pipeline classes. This is used to determine
            which pipeline class to instantiate based on the model type inferred from the checkpoint.
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

    # Load the checkpoint from the provided link or path
    checkpoint = load_single_file_checkpoint(pretrained_model_link_or_path)

    # Infer the model type from the loaded checkpoint
    model_type = infer_diffusers_model_type(checkpoint)

    # Get the corresponding pipeline class from the pipeline mapping
    pipeline_class = pipeline_mapping[model_type]

    # If the pipeline class is None, handle specific cases
    if pipeline_class is None:
        # Raise an error if the model type is in the inpaint pipeline keys
        if model_type in INPAINT_PIPELINE_KEYS:
            raise ValueError("`inpaint` is only supported in `AutoPipelineForInpainting.from_pretrained`")
        else:
            # Raise an error if the model type is not supported
            raise ValueError(f"{model_type} is not supported.")
    else:
        # Instantiate and return the pipeline with the loaded checkpoint and any additional kwargs
        return pipeline_class.from_single_file(pretrained_model_link_or_path, checkpoint=checkpoint, **kwargs)
