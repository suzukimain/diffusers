import os
import re
import json
import inspect
import importlib
import difflib
from dataclasses import (
    is_dataclass,
    dataclass,
    field
    )
from ..... import pipelines
from ....pipeline_utils import DiffusionPipeline
from .....loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    is_valid_url,
    _extract_repo_id_and_weights_name
    )
from .....utils.import_utils import is_natsort_available
from .pipeline_output import (
    DataConfig,
    RepoStatus,
    ModelStatus,
    SearchPipelineOutput
)

if is_natsort_available():
    from natsort import natsorted

TASK_KEY_MAPPING = {
    "txt2img": ["images"],
    "txt2video": ["frames"],
    "txt2audio": ["audios"],
}


CUSTOM_SEARCH_KEY = {
    "sd" : "stabilityai/stable-diffusion-2-1",
    }


CONFIG_FILE_LIST = [
    "preprocessor_config.json",
    "config.json",
    "model.fp16.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "scheduler_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
    "diffusion_pytorch_model.safetensors",
    "safety_checker/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.ckpt",
    "unet/diffusion_pytorch_model.ckpt",
    "vae/diffusion_pytorch_model.ckpt",
    "text_encoder/model.ckpt",
    "text_encoder/model.fp16.ckpt",
    "safety_checker/model.fp16.ckpt",
    "unet/diffusion_pytorch_model.fp16.ckpt",
    "vae/diffusion_pytorch_model.fp16.ckpt"
]

EXTENSION =  [".safetensors", ".ckpt",".bin"]

@dataclass
class DataConfig:
    """
    Configuration class for data handling in the model search pipeline.

    Attributes:
    - config_file: Path to the configuration file.
    - model_data: Dictionary containing model data.
    - model_info: Dictionary containing model information.
    - num_prints: Number of times to print the information.
    - force_download: Flag to force downloading the files.
    - single_file_only: Flag to handle only a single file.
    - hf_token: Hugging Face token for authentication.
    """
    config_file: str = "model_index.json"
    model_data: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    num_prints: int = 20
    force_download: bool = False
    single_file_only: bool = False
    hf_token: str = None



class SearchPipelineConfig(
    DiffusionPipeline,
    DataConfig
    ):
    """
    A class that provides configuration and utility methods for search pipelines.

    Methods:
        get_inherited_class(class_name): Returns a list of inherited classes.
        get_item(dict_obj): Returns the first element of the dictionary.
        pipe_class_type(class_name, skip_error=False): Determines the type of the pipeline class.
        old_pipe_class_type(class_name): [deprecated]Determines the type of the pipeline class.
        pipeline_metod_type(Target_class): Determines the method type of the pipeline.
        import_pipeline(class_name, skip_error=False): Imports a pipeline class from the `pipelines` module.
        get_func_method(target_class, method_name='__call__'): Returns the arguments of a function in a class.
        get_class_attributes(search): Get the names of the attributes defined in a class.
        check_for_safetensors(path): Checks if the file is a safetensors file.
        find_closest_match(search_word, search_list): Finds the closest match for a given search word in a list of words.
        filter_list_by_text(list_obj, need_txt): Filters a list of objects based on a text string.
        sort_by_version(sorted_list): Sorts a list by version in order of newest to oldest.
        get_pipeline_output_keys(class_obj): Returns a list of keys of the data class of the pipeline return value.
    """

    def __init__(self):
        super().__init__()


    def get_inherited_class(self, class_name) -> list:
        """
        Returns a list of inherited classes.

        Args:
            class_name (class): The class to get the inherited classes for.

        Returns:
            list: A list of inherited classes.
        """
        inherited_class = inspect.getmro(class_name)
        return [cls_method.__name__ for cls_method in inherited_class]

    def get_item(self, dict_obj):
        """
        Returns the first element of the dictionary.

        Args:
            dict_obj (dict): The dictionary to get the first element from.

        Returns:
            Any: The first element of the dictionary.
        """
        return next(iter(dict_obj.items()))[1]

    def pipe_class_type(self, class_name, skip_error=False):
        """
        Determines the type of the pipeline class.

        Args:
            class_name (class): The class to determine the type for.
            skip_error (bool, optional): Whether to skip errors. Defaults to False.

        Returns:
            str: The type of the pipeline class.

        Raises:
            ValueError: If the class is not supported.
        """
        class_output_list = self.get_pipeline_output_keys(class_name)
        for key, value in TASK_KEY_MAPPING.items():
            if key in class_output_list:
                return value
        else:
            if skip_error:
                return None
            else:
                raise ValueError(f"{class_name.__name__} is not supported")

    def old_pipe_class_type(self, class_name):
        """
        Determines the old type of the pipeline class.

        Args:
            class_name (class): The class to determine the old type for.

        Returns:
            str: The old type of the pipeline class.
        """
        _txt2img_method_list = []  # else
        _img2img_method_list = ["image"]
        _img2video_method_list = ["video_length", "fps"]

        call_method = self.get_call_method(class_name, method_name='__call__')

        if any(method in call_method for method in _img2video_method_list):
            pipeline_type = "txt2video"
        elif any(method in call_method for method in _img2img_method_list):
            pipeline_type = "img2img"
        else:
            pipeline_type = "txt2img"
        return pipeline_type

    def pipeline_metod_type(self, Target_class) -> str:
        """
        Determines the method type of the pipeline.

        Args:
            Target_class (class): The target class to determine the method type for.

        Returns:
            str: The method type of the pipeline.
        """
        torch_list = ["DiffusionPipeline",
                      "AutoPipelineForText2Image",
                      "AutoPipelineForImage2Image",
                      "AutoPipelineForInpainting",
                      ]

        flax_list = ["FlaxDiffusionPipeline", ]

        _class = self.import_pipeline(class_name=Target_class)

        cls_method = self.get_inherited_class(_class)

        if any(method in torch_list for method in cls_method):
            class_type = "torch"
        elif any(method in flax_list for method in cls_method):
            class_type = "flax"
        else:
            class_type = "onnx"
        return class_type

    def import_pipeline(self, class_name, skip_error: bool = False):
        """
        Import a pipeline class from the `pipelines` module.

        Args:
            class_name (str or class): The name of the pipeline class to import.
            skip_error (bool, optional): Whether to skip errors if the pipeline class is not found. Defaults to False.

        Returns:
            class: The imported pipeline class.

        Raises:
            Exception: If the class is not found and skip_error is False.
        """
        if isinstance(class_name, str):
            try:
                return getattr(pipelines, class_name)
            except Exception as err:
                if skip_error:
                    return None
                else:
                    error_txt = self.find_closest_match(
                        search_word=class_name,
                        search_list=dir(pipelines)
                    )
                    raise Exception(f"{class_name} not found. Maybe, {error_txt}?") from err
        else:
            return class_name

    def get_func_method(self, target_class, method_name: str = '__call__') -> list:
        """
        Returns the arguments of a function in a class.

        Args:
            target_class (class): The class containing the method.
            method_name (str, optional): The name of the method to get the arguments for. Defaults to '__call__'.

        Returns:
            list: A list of the argument names for the method.
        """
        func_name = getattr(target_class, method_name)
        parameters = inspect.signature(func_name).parameters
        arg_names = []
        for param in parameters.values():
            arg_names.append(param.name)
        return arg_names

    def get_class_attributes(self, search) -> list:
        """
        Get the names of the attributes defined in a class.

        Args:
            search (class): The class to search for attributes.

        Returns:
            list: A list of the names of the attributes defined in the class.
        """
        return list(search.__class__.__annotations__.keys())

    def check_for_safetensors(self, path):
        """
        Checks if the file is a safetensors file.

        Args:
            path (str): The path to the file.

        Returns:
            bool: True if the file is a safetensors file, False otherwise.
        """
        _ext = os.path.basename(path).split(".")[-1]
        if _ext == "safetensors":
            return True
        else:
            return False

    def find_closest_match(self, search_word: str, search_list: list) -> str:
        """
        Find the closest match for a given search word in a list of words.

        Args:
            search_word (str): The word to search for.
            search_list (list): The list of words to search in.

        Returns:
            str: The closest match for the search word.
        """
        return difflib.get_close_matches(search_word, search_list, cutoff=0, n=1)

    def filter_list_by_text(self, list_obj: list, need_txt: str) -> list:
        """
        Filter a list of objects based on a text string.

        Args:
            list_obj (list): The list of objects to filter.
            need_txt (str): The text string to search for.

        Returns:
            list: A new list containing only the objects that contain the text string.
        """
        sorted_list = []
        for module_obj in list_obj:
            if need_txt.lower() in module_obj.lower():
                sorted_list.append(module_obj)
        return sorted_list

    def sort_by_version(self, sorted_list) -> list:
        """
        Sorts a list by version in order of newest to oldest.

        Args:
            sorted_list (list): The list to sort.

        Returns:
            list: The sorted list.
        """
        return natsorted(sorted_list, reverse=True) if is_natsort_available() else sorted(sorted_list, reverse=True)

    def get_pipeline_output_keys(self, class_obj) -> list:
        """
        Returns a list of keys of the data class of the pipeline return value.

        Args:
            class_obj (class): The pipeline class object.

        Returns:
            list: A list of keys of the data class of the pipeline return value.
        """
        module_name = class_obj.__module__
        output_class_name = class_obj.__name__ + "Output"
        module = importlib.import_module(module_name)
        if hasattr(module, output_class_name):
            output_class = getattr(module, output_class_name)
            if is_dataclass(output_class):
                return list(output_class.__dataclass_fields__.keys())
        return []

    def get_keyword_types(
        self,
        keyword
    ):
        status = {
            "model_format": None,
            "loading_method": None,
            "type": {
                "search_word":False,
                "hf_url": False,
                "hf_repo": False,
                "civitai_url": False,
                "local": False,
                },
            "extra_type": {
                "url": False,
                "missing_model_index":None,
                },
            }
        
        status["extra_type"]["url"] = bool(re.match(r"^(https?)://", keyword))

        if os.path.isfile(keyword):
            status["type"]["local"] = True
            status["model_format"] = "single_file"
            status["loading_method"] = "from_single_file"

        elif os.path.isdir(keyword):
            status["type"]["local"] = True
            status["model_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"
            if not os.path.exists(os.path.join(keyword, "model_index.json")):
                status["extra_type"]["missing_model_index"] = True
        
        elif keyword.startswith("https://civitai.com/"):
            status["type"]["civitai_url"] = True
            status["model_format"] = "single_file"
            status["loading_method"] = None

        elif any(keyword.startswith(prefix) for prefix in VALID_URL_PREFIXES):
            repo_id,weights_name = _extract_repo_id_and_weights_name(keyword)
            if weights_name:
                status["type"]["hf_url"] = True
                status["model_format"] = "single_file"
                status["loading_method"] = "from_single_file"
            else:
                status["type"]["hf_repo"] = True
                status["model_format"] = "diffusers"
                status["loading_method"] = "from_pretrained"

        elif re.match(r"^[^/]+/[^/]+$", keyword):
            status["type"]["hf_repo"] = True
            status["model_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"
        
        else:
            status["type"]["search_word"] = True
            status["model_format"] = None
            status["loading_method"] = None
        
        return status

    def scan_dir(self, path):
        ext = ['.safetensors']
        for entry in os.scandir(path):
            if entry.is_dir():
                yield from self.scan_dir(entry.path)
            elif entry.is_file() and entry.name.endswith(tuple(ext)):
                yield os.path.abspath(entry.path)    
