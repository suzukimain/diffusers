import os
import re
import requests
from typing import (
    Union,
    List
)
from tqdm.auto import tqdm
from dataclasses import (
    asdict,
    dataclass
)
from huggingface_hub import (
    hf_api,
    hf_hub_download,
)

from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    _extract_repo_id_and_weights_name,
)


CONFIG_FILE_LIST = [
    "preprocessor_config.json",
    "config.json",
    "model.safetensors",
    "model.fp16.safetensors",
    "model.ckpt",
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "scheduler_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.ckpt",
    "diffusion_pytorch_model.fp16.ckpt",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
    "safety_checker/pytorch_model.bin",
    "safety_checker/pytorch_model.fp16.bin",
    "safety_checker/model.safetensors",
    "safety_checker/model.ckpt",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.fp16.ckpt",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.fp16.bin",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.ckpt",
    "unet/diffusion_pytorch_model.fp16.ckpt",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.bin",
    "vae/diffusion_pytorch_model.ckpt",
    "vae/diffusion_pytorch_model.fp16.ckpt",
    "text_encoder/pytorch_model.bin",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.ckpt",
    "text_encoder/model.fp16.ckpt",
    "text_encoder_2/model.safetensors",
    "text_encoder_2/model.ckpt"
]


CONFIG_FILE_LIST = [
    "model.safetensors",
    "model.fp16.safetensors",
    "model.ckpt",
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.ckpt",
    "diffusion_pytorch_model.fp16.ckpt",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
    "safety_checker/pytorch_model.bin",
    "safety_checker/pytorch_model.fp16.bin",
    "safety_checker/model.safetensors",
    "safety_checker/model.ckpt",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.fp16.ckpt",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.fp16.bin",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.ckpt",
    "unet/diffusion_pytorch_model.fp16.ckpt",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.bin",
    "vae/diffusion_pytorch_model.ckpt",
    "vae/diffusion_pytorch_model.fp16.ckpt",
    "text_encoder/pytorch_model.bin",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.ckpt",
    "text_encoder/model.fp16.ckpt",
    "text_encoder_2/model.safetensors",
    "text_encoder_2/model.ckpt"
]





EXTENSION =  [".safetensors", ".ckpt",".bin"]


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



@dataclass
class RepoStatus:
    r"""
    Data class for storing repository status information.

    Attributes:
        repo_id (`str`):
            The name of the repository.
        repo_hash (`str`):
            The hash of the repository.
        version (`str`):
            The version ID of the repository.
    """
    repo_id: str = ""
    repo_hash: str = ""
    version: str = ""

@dataclass
class ModelStatus:
    r"""
    Data class for storing model status information.

    Attributes:
        search_word (`str`):
            The search word used to find the model.
        download_url (`str`):
            The URL to download the model.
        file_name (`str`):
            The name of the model file.
        local (`bool`):
            Whether the model exists locally
    """
    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False


@dataclass
class PipelineSearchResult:
    r"""
    Data class for storing model data.

    Attributes:
        model_path (`str`):
            The path to the model.
        loading_method (`str`):
            The type of loading method used for the model ( None or 'from_single_file' or 'from_pretrained')
        checkpoint_format (`str`):
            The format of the model checkpoint (`single_file` or `diffusers`).
        repo_status (`RepoStatus`):
            The status of the repository.
        model_status (`ModelStatus`):
            The status of the model.
    """
    model_path: str = ""
    loading_method: str = None  
    checkpoint_format: str = None
    repo_status: RepoStatus = RepoStatus()
    model_status: ModelStatus = ModelStatus()
    


def get_keyword_types(keyword):
    r"""
    Determine the type and loading method for a given keyword.

    Parameters:
        keyword (`str`):
            The input keyword to classify.

    Returns:
        `dict`: A dictionary containing the model format, loading method,
                and various types and extra types flags.
    """
    
    # Initialize the status dictionary with default values
    status = {
        "checkpoint_format": None,
        "loading_method": None,
        "type": {
            "other": False,
            "hf_url": False,
            "hf_repo": False,
            "civitai_url": False,
            "local": False,
        },
        "extra_type": {
            "url": False,
            "missing_model_index": None,
        },
    }
    
    # Check if the keyword is an HTTP or HTTPS URL
    status["extra_type"]["url"] = bool(re.search(r"^(https?)://", keyword))
    
    # Check if the keyword is a file
    if os.path.isfile(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = "from_single_file"
    
    # Check if the keyword is a directory
    elif os.path.isdir(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
        if not os.path.exists(os.path.join(keyword, "model_index.json")):
            status["extra_type"]["missing_model_index"] = True
    
    # Check if the keyword is a Civitai URL
    elif keyword.startswith("https://civitai.com/"):
        status["type"]["civitai_url"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = None
    
    # Check if the keyword starts with any valid URL prefixes
    elif any(keyword.startswith(prefix) for prefix in VALID_URL_PREFIXES):
        repo_id, weights_name = _extract_repo_id_and_weights_name(keyword)
        if weights_name:
            status["type"]["hf_url"] = True
            status["checkpoint_format"] = "single_file"
            status["loading_method"] = "from_single_file"
        else:
            status["type"]["hf_repo"] = True
            status["checkpoint_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"
    
    # Check if the keyword matches a Hugging Face repository format
    elif re.match(r"^[^/]+/[^/]+$", keyword):
        status["type"]["hf_repo"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
    
    # If none of the above apply
    else:
        status["type"]["other"] = True
        status["checkpoint_format"] = None
        status["loading_method"] = None
    
    return status


class HFSearchPipeline:
    """
    Search for models from Huggingface.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def create_huggingface_url(repo_id, file_name):
        r"""
        Create a Hugging Face URL for a given repository ID and file name.

        Parameters:
            repo_id (`str`):
                The repository ID.
            file_name (`str`):
                The file name within the repository.

        Returns:
            `str`: The complete URL to the file or repository on Hugging Face.
        """
        if file_name:
            return f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        else:
            return f"https://huggingface.co/{repo_id}"
    
    @staticmethod
    def hf_find_safest_model(models) -> str:
        r"""
        Sort and find the safest model.

        Parameters:
            models (`list`):
                A list of model names to sort and check.

        Returns:
            `str`: The name of the safest model or the first model in the list if no safe model is found.
        """
        for model in sorted(models, reverse=True):
            if bool(re.search(r"(?i)[-_](safe|sfw)", model)):
                return model
        return models[0]
    
    @classmethod
    def for_HF(cls, search_word: str, **kwargs) -> Union[str, SearchPipelineOutput, None]:
        r"""
        Downloads a model from Hugging Face.

        Parameters:
            search_word (`str`):
                The search query string.
            revision (`str`, *optional*):
                The specific version of the model to download.
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            download (`bool`, *optional*, defaults to `False`):
                Whether to download the model.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force the download if the model already exists.
            include_params (`bool`, *optional*, defaults to `False`):
                Whether to include parameters in the returned data.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            hf_token (`str`, *optional*):
                API token for Hugging Face authentication.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str, SearchPipelineOutput, None]`: The model path or SearchPipelineOutput or None.
        """
        # Extract additional parameters from kwargs
        revision = kwargs.pop("revision", None)
        checkpoint_format = kwargs.pop("checkpoint_format", "single_file")
        download = kwargs.pop("download", False)
        force_download = kwargs.pop("force_download", False)
        include_params = kwargs.pop("include_params", False)
        pipeline_tag = kwargs.pop("pipeline_tag", None)
        hf_token = kwargs.pop("hf_token", None)
        skip_error = kwargs.pop("skip_error", False)

        # Get the type and loading method for the keyword
        search_word_status = get_keyword_types(search_word)

        if search_word_status["type"]["hf_repo"]:
            if download:
                model_path = DiffusionPipeline.download(
                    search_word,
                    revision=revision,
                    token=hf_token
                )
            else:
                model_path = search_word
        elif search_word_status["type"]["hf_url"]:
            repo_id, weights_name = _extract_repo_id_and_weights_name(search_word)
            if download:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=weights_name,
                    force_download=force_download,
                    token=hf_token
                )
            else:
                model_path = search_word
        elif search_word_status["type"]["local"]:
            model_path = search_word
        elif search_word_status["type"]["civitai_url"]:
            if skip_error:
                return None
            else:
                raise ValueError("The URL for Civitai is invalid with `for_hf`. Please use `for_civitai` instead.")
        
        else:
            # Get model data from HF API
            hf_models = hf_api.list_models(
                search=search_word,
                sort="downloads",
                direction=-1,
                limit=100,
                fetch_config=True,
                pipeline_tag=pipeline_tag,
                full=True,
                token=hf_token
            )
            model_dicts = [asdict(value) for value in list(hf_models)]
            
            hf_repo_info = {}
            file_list = []
            repo_id, file_name = "", ""
            diffusers_model_exists = False

            # Loop through models to find a suitable candidate
            for repo_info in model_dicts:
                repo_id = repo_info["id"]
                file_list = []
                hf_repo_info = hf_api.model_info(
                    repo_id=repo_id,
                    securityStatus=True
                )
                # Lists files with security issues.
                hf_security_info = hf_repo_info.security_repo_status
                exclusion = [issue['path'] for issue in hf_security_info['filesWithIssues']]

                # Checks for multi-folder diffusers model or valid files (models with security issues are excluded).
                if hf_security_info["scansDone"]:
                    for info in repo_info["siblings"]:
                        file_path = info["rfilename"]
                        if (
                            "model_index.json" == file_path
                            and checkpoint_format in ["diffusers", "all"]
                        ):
                            diffusers_model_exists = True
                            break
                        
                        elif (
                            any(file_path.endswith(ext) for ext in EXTENSION)
                            and not any(config in file_path for config in CONFIG_FILE_LIST)
                            and not any(exc in file_path for exc in exclusion)
                        ):
                            file_list.append(file_path)
                
                # Exit from the loop if a multi-folder diffusers model or valid file is found
                if diffusers_model_exists or file_list:
                    break
            else:
                # Handle case where no models match the criteria
                if skip_error:
                    return None
                else:
                    raise ValueError("No models matching your criteria were found on huggingface.")
            
            download_url = cls.create_huggingface_url(
                repo_id=repo_id, file_name=file_name
            )
            if diffusers_model_exists:
                if download:
                    model_path = DiffusionPipeline.download(
                        repo_id=repo_id,
                        token=hf_token,
                    )
                else:
                    model_path = repo_id
                    
            elif file_list:
                file_name = cls.hf_find_safest_model(file_list)
                if download:
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_name,
                        revision=revision,
                        token=hf_token
                    )
                else:
                    model_path = cls.create_huggingface_url(
                        repo_id=repo_id, file_name=file_name
                    )
        
        output_info = get_keyword_types(model_path)

        if include_params:
            return SearchPipelineOutput(
                model_path=model_path,
                loading_method=output_info["loading_method"],
                checkpoint_format=output_info["checkpoint_format"],
                repo_status=RepoStatus(
                    repo_id=repo_id,
                    repo_hash=hf_repo_info.sha,
                    version=revision
                ),
                model_status=ModelStatus(
                    search_word=search_word,
                    download_url=download_url,
                    file_name=file_name,
                    local=download,
                )
            )
        
        else:
            return model_path    



class CivitaiSearchPipeline:
    """
    Find checkpoints and more from Civitai.
    """

    def __init__(self):
        pass

    @staticmethod
    def civitai_find_safest_model(models: List[dict]) -> dict:
        r"""
        Sort and find the safest model.
        
        Parameters:
            models (`list`):
                A list of model dictionaries to check. Each dictionary should contain a 'filename' key.
        
        Returns:
            `dict`: The dictionary of the safest model or the first model in the list if no safe model is found.
        """
        
        for model_data in models:
            if bool(re.search(r"(?i)[-_](safe|sfw)", model_data["filename"])):
                return model_data
        return models[0]

    @classmethod
    def for_civitai(
        cls,
        search_word: str,
        **kwargs
    ) -> Union[str, SearchPipelineOutput, None]:
        r"""
        Downloads a model from Civitai.

        Parameters:
            search_word (`str`):
                The search query string.
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for.
            base_model (`str`, *optional*):
                The base model to filter by.
            download (`bool`, *optional*, defaults to `False`):
                Whether to download the model.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force the download if the model already exists.
            civitai_token (`str`, *optional*):
                API token for Civitai authentication.
            include_params (`bool`, *optional*, defaults to `False`):
                Whether to include parameters in the returned data.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str, SearchPipelineOutput, None]`: The model path or `SearchPipelineOutput` or None.
        """

        # Extract additional parameters from kwargs
        model_type = kwargs.pop("model_type", "Checkpoint")
        download = kwargs.pop("download", False)
        base_model = kwargs.pop("base_model", None)
        force_download = kwargs.pop("force_download", False)
        civitai_token = kwargs.pop("civitai_token", None)
        include_params = kwargs.pop("include_params", False)
        skip_error = kwargs.pop("skip_error", False)

        # Initialize additional variables with default values
        model_path = ""
        repo_name = ""
        repo_id = ""
        version_id = ""
        models_list = []
        selected_repo = {}
        selected_model = {}
        selected_version = {}

        # Set up parameters and headers for the CivitAI API request
        params = {
            "query": search_word,
            "types": model_type,
            "sort": "Highest Rated",
            "limit": 20
        }
        if base_model is not None:
            params["baseModel"] = base_model

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

        try:
            # Make the request to the CivitAI API
            response = requests.get(
                "https://civitai.com/api/v1/models", params=params, headers=headers
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise requests.HTTPError(f"Could not get elements from the URL: {err}")
        else:
            try:
                data = response.json()
            except AttributeError:
                if skip_error:
                    return None
                else:
                    raise ValueError("Invalid JSON response")
        
        # Sort repositories by download count in descending order
        sorted_repos = sorted(data["items"], key=lambda x: x["stats"]["downloadCount"], reverse=True)

        for selected_repo in sorted_repos:
            repo_name = selected_repo["name"]
            repo_id = selected_repo["id"]

            # Sort versions within the selected repo by download count
            sorted_versions = sorted(selected_repo["modelVersions"], key=lambda x: x["stats"]["downloadCount"], reverse=True)
            for selected_version in sorted_versions:
                version_id = selected_version["id"]
                models_list = []
                for model_data in selected_version["files"]:
                    # Check if the file passes security scans and has a valid extension
                    if (
                        model_data["pickleScanResult"] == "Success"
                        and model_data["virusScanResult"] == "Success"
                        and any(model_data["name"].endswith(ext) for ext in EXTENSION)
                    ):
                        file_status = {
                            "filename": model_data["name"],
                            "download_url": model_data["downloadUrl"],
                        }
                        models_list.append(file_status)

                if models_list:
                    # Sort the models list by filename and find the safest model
                    sorted_models = sorted(models_list, key=lambda x: x["filename"], reverse=True)
                    selected_model = cls.civitai_find_safest_model(sorted_models)
                    break
            else:
                continue
            break

        if not selected_model:
            if skip_error:
                return None
            else:
                raise ValueError("No model found. Please try changing the word you are searching for.")

        file_name = selected_model["filename"]
        download_url = selected_model["download_url"]

        # Handle file download and setting model information
        if download:
            model_path = f"/root/.cache/Civitai/{repo_id}/{version_id}/{file_name}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if (not os.path.exists(model_path)) or force_download:
                headers = {}
                if civitai_token:
                    headers["Authorization"] = f"Bearer {civitai_token}"

                try:
                    response = requests.get(download_url, stream=True, headers=headers)
                    response.raise_for_status()
                except requests.HTTPError:
                    raise requests.HTTPError(f"Invalid URL: {download_url}, {response.status_code}")

                with tqdm.wrapattr(
                    open(model_path, "wb"),
                    "write",
                    miniters=1,
                    desc=file_name,
                    total=int(response.headers.get("content-length", 0)),
                ) as fetched_model_info:
                    for chunk in response.iter_content(chunk_size=8192):
                        fetched_model_info.write(chunk)
        else:
            model_path = download_url

        output_info = get_keyword_types(model_path)

        if not include_params:
            return model_path
        else:
            return SearchPipelineOutput(
                model_path=model_path,
                loading_method=output_info["loading_method"],
                checkpoint_format=output_info["checkpoint_format"],
                repo_status=RepoStatus(
                    repo_id=repo_name,
                    repo_hash=repo_id,
                    version=version_id
                ),
                model_status=ModelStatus(
                    search_word=search_word,
                    download_url=download_url,
                    file_name=file_name,
                    local=output_info["type"]["local"]
                )
            )



class ModelSearchPipeline(
    HFSearchPipeline,
    CivitaiSearchPipeline
    ):

    def __init__(self):
        pass
    
    @classmethod
    def for_hubs(
        cls,
        search_word: str,
        **kwargs
    ) -> Union[None, str, SearchPipelineOutput]:
        r"""
        Search and download models from multiple hubs.

        Parameters:
            search_word (`str`):
                The search query string.
            model_type (`str`, *optional*, defaults to `Checkpoint`, Civitai only):
                The type of model to search for.
            revision (`str`, *optional*, Hugging Face only):
                The specific version of the model to download.
            include_params (`bool`, *optional*, defaults to `False`, both):
                Whether to include parameters in the returned data.
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`, Hugging Face only):
                The format of the model checkpoint.
            download (`bool`, *optional*, defaults to `False`, both):
                Whether to download the model.
            pipeline_tag (`str`, *optional*, Hugging Face only):
                Tag to filter models by pipeline.
            base_model (`str`, *optional*, Civitai only):
                The base model to filter by.
            force_download (`bool`, *optional*, defaults to `False`, both):
                Whether to force the download if the model already exists.  
            hf_token (`str`, *optional*, Hugging Face only):
                API token for Hugging Face authentication.
            civitai_token (`str`, *optional*, Civitai only):
                API token for Civitai authentication.
            skip_error (`bool`, *optional*, defaults to `False`, both):
                Whether to skip errors and return None.

        Returns:
            `Union[None, str, SearchPipelineOutput]`: The model path, SearchPipelineOutput, or None if not found.
        """

        return (
            cls.for_HF(search_word, skip_error=True, **kwargs) 
            or cls.for_civitai(search_word, skip_error=True, **kwargs)
        )
    








# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import os

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, validate_hf_hub_args
from packaging import version

from ..utils import deprecate, is_transformers_available, logging
from .single_file_utils import (
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


logger = logging.get_logger(__name__)

# Legacy behaviour. `from_single_file` does not load the safety checker unless explicitly provided
SINGLE_FILE_OPTIONAL_COMPONENTS = ["safety_checker"]

if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_single_file_sub_model(
    library_name,
    class_name,
    name,
    checkpoint,
    pipelines,
    is_pipeline_module,
    cached_model_config_path,
    original_config=None,
    local_files_only=False,
    torch_dtype=None,
    is_legacy_loading=False,
    **kwargs,
):
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )
    is_tokenizer = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedTokenizer)
        and transformers_version >= version.parse("4.20.0")
    )

    diffusers_module = importlib.import_module(__name__.split(".")[0])
    is_diffusers_single_file_model = issubclass(class_obj, diffusers_module.FromOriginalModelMixin)
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    is_diffusers_scheduler = issubclass(class_obj, diffusers_module.SchedulerMixin)

    if is_diffusers_single_file_model:
        load_method = getattr(class_obj, "from_single_file")

        # We cannot provide two different config options to the `from_single_file` method
        # Here we have to ignore loading the config from `cached_model_config_path` if `original_config` is provided
        if original_config:
            cached_model_config_path = None

        loaded_sub_model = load_method(
            pretrained_model_link_or_path_or_dict=checkpoint,
            original_config=original_config,
            config=cached_model_config_path,
            subfolder=name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )

    elif is_transformers_model and is_clip_model_in_single_file(class_obj, checkpoint):
        loaded_sub_model = create_diffusers_clip_model_from_ldm(
            class_obj,
            checkpoint=checkpoint,
            config=cached_model_config_path,
            subfolder=name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            is_legacy_loading=is_legacy_loading,
        )

    elif is_transformers_model and is_t5_in_single_file(checkpoint):
        loaded_sub_model = create_diffusers_t5_model_from_checkpoint(
            class_obj,
            checkpoint=checkpoint,
            config=cached_model_config_path,
            subfolder=name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    elif is_tokenizer and is_legacy_loading:
        loaded_sub_model = _legacy_load_clip_tokenizer(
            class_obj, checkpoint=checkpoint, config=cached_model_config_path, local_files_only=local_files_only
        )

    elif is_diffusers_scheduler and (is_legacy_loading or _is_legacy_scheduler_kwargs(kwargs)):
        loaded_sub_model = _legacy_load_scheduler(
            class_obj, checkpoint=checkpoint, component_name=name, original_config=original_config, **kwargs
        )

    else:
        if not hasattr(class_obj, "from_pretrained"):
            raise ValueError(
                (
                    f"The component {class_obj.__name__} cannot be loaded as it does not seem to have"
                    " a supported loading method."
                )
            )

        loading_kwargs = {}
        loading_kwargs.update(
            {
                "pretrained_model_name_or_path": cached_model_config_path,
                "subfolder": name,
                "local_files_only": local_files_only,
            }
        )

        # Schedulers and Tokenizers don't make use of torch_dtype
        # Skip passing it to those objects
        if issubclass(class_obj, torch.nn.Module):
            loading_kwargs.update({"torch_dtype": torch_dtype})

        if is_diffusers_model or is_transformers_model:
            if not _is_model_weights_in_cached_folder(cached_model_config_path, name):
                raise SingleFileComponentError(
                    f"Failed to load {class_name}. Weights for this component appear to be missing in the checkpoint."
                )

        load_method = getattr(class_obj, "from_pretrained")
        loaded_sub_model = load_method(**loading_kwargs)

    return loaded_sub_model


def _map_component_types_to_config_dict(component_types):
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    config_dict = {}
    component_types.pop("self", None)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    for component_name, component_value in component_types.items():
        is_diffusers_model = issubclass(component_value[0], diffusers_module.ModelMixin)
        is_scheduler_enum = component_value[0].__name__ == "KarrasDiffusionSchedulers"
        is_scheduler = issubclass(component_value[0], diffusers_module.SchedulerMixin)

        is_transformers_model = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedModel)
            and transformers_version >= version.parse("4.20.0")
        )
        is_transformers_tokenizer = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedTokenizer)
            and transformers_version >= version.parse("4.20.0")
        )

        if is_diffusers_model and component_name not in SINGLE_FILE_OPTIONAL_COMPONENTS:
            config_dict[component_name] = ["diffusers", component_value[0].__name__]

        elif is_scheduler_enum or is_scheduler:
            if is_scheduler_enum:
                # Since we cannot fetch a scheduler config from the hub, we default to DDIMScheduler
                # if the type hint is a KarrassDiffusionSchedulers enum
                config_dict[component_name] = ["diffusers", "DDIMScheduler"]

            elif is_scheduler:
                config_dict[component_name] = ["diffusers", component_value[0].__name__]

        elif (
            is_transformers_model or is_transformers_tokenizer
        ) and component_name not in SINGLE_FILE_OPTIONAL_COMPONENTS:
            config_dict[component_name] = ["transformers", component_value[0].__name__]

        else:
            config_dict[component_name] = [None, None]

    return config_dict


def _infer_pipeline_config_dict(pipeline_class):
    parameters = inspect.signature(pipeline_class.__init__).parameters
    required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
    component_types = pipeline_class._get_signature_types()

    # Ignore parameters that are not required for the pipeline
    component_types = {k: v for k, v in component_types.items() if k in required_parameters}
    config_dict = _map_component_types_to_config_dict(component_types)

    return config_dict


def _download_diffusers_model_config_from_hub(
    pretrained_model_name_or_path,
    cache_dir,
    revision,
    proxies,
    force_download=None,
    local_files_only=None,
    token=None,
):
    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]
    cached_model_path = snapshot_download(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        proxies=proxies,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=allow_patterns,
    )

    return cached_model_path


class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):

        original_config_file = kwargs.pop("original_config_file", None)
        config = kwargs.pop("config", None)
        original_config = kwargs.pop("original_config", None)

        if original_config_file is not None:
            deprecation_message = (
                "`original_config_file` argument is deprecated and will be removed in future versions."
                "please use the `original_config` argument instead."
            )
            deprecate("original_config_file", "1.0.0", deprecation_message)
            original_config = original_config_file

        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        is_legacy_loading = False

        # We shouldn't allow configuring individual models components through a Pipeline creation method
        # These model kwargs should be deprecated
        scaling_factor = kwargs.get("scaling_factor", None)
        if scaling_factor is not None:
            deprecation_message = (
                "Passing the `scaling_factor` argument to `from_single_file is deprecated "
                "and will be ignored in future versions."
            )
            deprecate("scaling_factor", "1.0.0", deprecation_message)

        if original_config is not None:
            original_config = fetch_original_config(original_config, local_files_only=local_files_only)

        from ..pipelines.pipeline_utils import _get_pipeline_class

        pipeline_class = _get_pipeline_class(cls, config=None)

        checkpoint = load_single_file_checkpoint(
            pretrained_model_link_or_path,
            force_download=force_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        if config is None:
            config = fetch_diffusers_config(checkpoint)
            default_pretrained_model_config_name = config["pretrained_model_name_or_path"]
        else:
            default_pretrained_model_config_name = config

        if not os.path.isdir(default_pretrained_model_config_name):
            # Provided config is a repo_id
            if default_pretrained_model_config_name.count("/") > 1:
                raise ValueError(
                    f'The provided config "{config}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            try:
                # Attempt to download the config files for the pipeline
                cached_model_config_path = _download_diffusers_model_config_from_hub(
                    default_pretrained_model_config_name,
                    cache_dir=cache_dir,
                    revision=revision,
                    proxies=proxies,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                )
                config_dict = pipeline_class.load_config(cached_model_config_path)

            except LocalEntryNotFoundError:
                # `local_files_only=True` but a local diffusers format model config is not available in the cache
                # If `original_config` is not provided, we need override `local_files_only` to False
                # to fetch the config files from the hub so that we have a way
                # to configure the pipeline components.

                if original_config is None:
                    logger.warning(
                        "`local_files_only` is True but no local configs were found for this checkpoint.\n"
                        "Attempting to download the necessary config files for this pipeline.\n"
                    )
                    cached_model_config_path = _download_diffusers_model_config_from_hub(
                        default_pretrained_model_config_name,
                        cache_dir=cache_dir,
                        revision=revision,
                        proxies=proxies,
                        force_download=force_download,
                        local_files_only=False,
                        token=token,
                    )
                    config_dict = pipeline_class.load_config(cached_model_config_path)

                else:
                    # For backwards compatibility
                    # If `original_config` is provided, then we need to assume we are using legacy loading for pipeline components
                    logger.warning(
                        "Detected legacy `from_single_file` loading behavior. Attempting to create the pipeline based on inferred components.\n"
                        "This may lead to errors if the model components are not correctly inferred. \n"
                        "To avoid this warning, please explicity pass the `config` argument to `from_single_file` with a path to a local diffusers model repo \n"
                        "e.g. `from_single_file(<my model checkpoint path>, config=<path to local diffusers model repo>) \n"
                        "or run `from_single_file` with `local_files_only=False` first to update the local cache directory with "
                        "the necessary config files.\n"
                    )
                    is_legacy_loading = True
                    cached_model_config_path = None

                    config_dict = _infer_pipeline_config_dict(pipeline_class)
                    config_dict["_class_name"] = pipeline_class.__name__

        else:
            # Provided config is a path to a local directory attempt to load directly.
            cached_model_config_path = default_pretrained_model_config_name
            config_dict = pipeline_class.load_config(cached_model_config_path)

        #   pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        expected_modules, optional_kwargs = pipeline_class._get_signature_keys(cls)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        from diffusers import pipelines

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            if name in SINGLE_FILE_OPTIONAL_COMPONENTS:
                return False

            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        for name, (library_name, class_name) in logging.tqdm(
            sorted(init_dict.items()), desc="Loading pipeline components..."
        ):
            loaded_sub_model = None
            is_pipeline_module = hasattr(pipelines, library_name)

            if name in passed_class_obj:
                loaded_sub_model = passed_class_obj[name]

            else:
                try:
                    loaded_sub_model = load_single_file_sub_model(
                        library_name=library_name,
                        class_name=class_name,
                        name=name,
                        checkpoint=checkpoint,
                        is_pipeline_module=is_pipeline_module,
                        cached_model_config_path=cached_model_config_path,
                        pipelines=pipelines,
                        torch_dtype=torch_dtype,
                        original_config=original_config,
                        local_files_only=local_files_only,
                        is_legacy_loading=is_legacy_loading,
                        **kwargs,
                    )
                except SingleFileComponentError as e:
                    raise SingleFileComponentError(
                        (
                            f"{e.message}\n"
                            f"Please load the component before passing it in as an argument to `from_single_file`.\n"
                            f"\n"
                            f"{name} = {class_name}.from_pretrained('...')\n"
                            f"pipe = {pipeline_class.__name__}.from_single_file(<checkpoint path>, {name}={name})\n"
                            f"\n"
                        )
                    )

            init_kwargs[name] = loaded_sub_model

        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components

        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # deprecated kwargs
        load_safety_checker = kwargs.pop("load_safety_checker", None)
        if load_safety_checker is not None:
            deprecation_message = (
                "Please pass instances of `StableDiffusionSafetyChecker` and `AutoImageProcessor`"
                "using the `safety_checker` and `feature_extractor` arguments in `from_single_file`"
            )
            deprecate("load_safety_checker", "1.0.0", deprecation_message)

            safety_checker_components = _legacy_load_safety_checker(local_files_only, torch_dtype)
            init_kwargs.update(safety_checker_components)

        pipe = pipeline_class(**init_kwargs)

        return pipe

