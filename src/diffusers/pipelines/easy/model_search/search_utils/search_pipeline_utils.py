import os
import json
import inspect
import importlib
import difflib
from dataclasses import is_dataclass

from ..... import pipelines
from .....utils.import_utils import is_natsort_available
from .model_search_data_classes import DataConfig

if is_natsort_available():
    from natsort import natsorted

TASK_KEY_MAPPING = {
    "txt2img": ["images"],
    "txt2video": ["frames"],
    "txt2audio": ["audios"],
}


class DataStoreManager:
    """
    A class to manage the storage and retrieval of data in a JSON file.

    Attributes:
        base_config_json (str): The path to the JSON config file.

    Methods:
        check_func_hist(key, **kwargs): Check and optionally update the history of a given element.
        get_json_dict(): Retrieve the JSON dictionary from the config file.
        update_json_dict(key, value): Update the JSON dictionary with a new key-value pair.
    """
    base_config_json = "/tmp/diffusers_easy_pipeline_config.json"

    def __init__(self):
        pass

    def check_func_hist(self, key, **kwargs):
        """
        Check and optionally update the history of a given element.

        Args:
            key (str): Specific key to look up in the dictionary.
            **kwargs: Keyword arguments for additional options.
                - update (bool): Whether to update the dictionary. Default is True.
                - return_value (bool): Whether to return the element value. Default is False.
                - value (Any): Value to be matched or updated in the dictionary.
                - missing_value (Any): Returns the value if it does not exist.

        Returns:
            Any: The historical value if `return_value` is True, or a boolean indicating
                 if the value matches the historical value.
        """
        value = kwargs.pop("value", None)
        update = kwargs.pop("update", False if value is None else True)
        return_value = kwargs.pop("return_value", True if "value" in kwargs else False)
        missing_value = kwargs.pop("missing_value", None)

        hist_value = self.get_json_dict().get(key, None)
        if update:
            self.update_json_dict(key, value)

        if return_value:
            return hist_value or missing_value
        else:
            return hist_value == value

    def get_json_dict(self) -> dict:
        """
        Retrieve the JSON dictionary from the config file.

        Returns:
            dict: The JSON dictionary.
        """
        config_dict = {}
        if os.path.isfile(self.base_config_json):
            try:
                with open(self.base_config_json, "r") as basic_json:
                    config_dict = json.load(basic_json)
            except json.JSONDecodeError:
                pass
        return config_dict

    def update_json_dict(self, key, value):
        """
        Update the JSON dictionary with a new key-value pair.

        Args:
            key (str): The key to update.
            value (Any): The value to associate with the key.
        """
        basic_json_dict = self.get_json_dict()
        basic_json_dict[key] = value
        with open(self.base_config_json, "w") as json_file:
            json.dump(basic_json_dict, json_file, indent=4)


class SearchPipelineConfig(DataConfig, DataStoreManager):
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
        self.device_count = self.count_device()
        self.device = self.device_type_check()

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