import os
import json
import inspect
import difflib

from natsort import natsorted

from .data_class import DataConfig


class DataStoreManager:
    base_config_json = "/tmp/diffusers_easy_pipeline_config.json"
    def __init__(self):
        pass

    def check_func_hist(
            self,
            key,
            **kwargs
            ):
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
        value = kwargs.pop("value",None)
        update = kwargs.pop("update", False if value is None else True)
        return_value = kwargs.pop("return_value", True if "value" in kwargs else False)
        missing_value = kwargs.pop("missing_value", None)

        hist_value = self.get_json_dict().get(key,None)
        if update:
            self.update_json_dict(key, value)

        if return_value:
            return hist_value or missing_value
        else:
            return hist_value == value


    def get_json_dict(self) -> dict:
        """
        Retrieve the JSON dictionary from the config file.
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
        """Update the JSON dictionary with a new key-value pair."""
        basic_json_dict = self.get_json_dict()
        basic_json_dict[key] = value
        with open(self.base_config_json, "w") as json_file:
            json.dump(basic_json_dict, json_file, indent=4)



class Basic_config(  
    DataConfig,
    DataStoreManager,
    Runtime_func,
    device_set
    ):

    def __init__(self):
        super().__init__()
        self.device_count = self.count_device()
        self.device = self.device_type_check()
        self.logger = custom_logger()


    @classmethod
    def get_inherited_class(cls,class_name) -> list:
        inherited_class = inspect.getmro(class_name)
        return [cls_method.__name__ for cls_method in inherited_class]


    def get_item(self,dict_obj):
        """
        Returns the first element of the dictionary
        """
        return next(iter(dict_obj.items()))[1]


    def pipe_class_type(
            self,
            class_name
            ):
        """
        Args:
        class_name : class

        Returns:
        Literal['txt2img','img2img','txt2video']
        """
        _txt2img_method_list = [] #else
        _img2img_method_list = ["image"]
        _img2video_method_list = ["video_length","fps"]

        call_method = self.get_call_method(class_name,method_name = '__call__')

        if any(method in call_method for method in _img2video_method_list):
            pipeline_type = "txt2video"
        elif any(method in call_method for method in _img2img_method_list):
            pipeline_type = "img2img"
        else:
            pipeline_type = "txt2img"
        return pipeline_type


    def pipeline_metod_type(self,Target_class) -> str:
        """
        Args:
        Target_class : class

        Returns:
        Literal['torch','flax','onnx']
        """
        torch_list=["DiffusionPipeline",
                    "AutoPipelineForText2Image",
                    "AutoPipelineForImage2Image",
                    "AutoPipelineForInpainting",]

        flax_list = ["FlaxDiffusionPipeline",]

        if isinstance(Target_class,str):
            Target_class = getattr(diffusers, Target_class)

        cls_method= self.get_inherited_class(Target_class)

        if any(method in torch_list for method in cls_method):
            class_type= "torch"
        elif any(method in flax_list for method in cls_method):
            class_type= "flax"
        else:
            class_type= "onnx"
        return class_type


    def get_call_method(
            self,
            class_name,
            method_name : str = '__call__'
            ) ->list:
        """
        Acquire the arguments of the function specified by 'method_name'
        for the class specified by 'class_name'
        """
        if isinstance(class_name,str):
            class_name = getattr(getattr(diffusers, class_name),method_name)
        parameters = inspect.signature(class_name).parameters
        arg_names = []
        for param in parameters.values():
            arg_names.append(param.name)
        return arg_names


    def get_class_elements(
            self,
            search
            ):
        return list(search.__class__.__annotations__.keys())


    def check_for_safetensors(
            self,
            path
            ):
        _ext = os.path.basename(path).split(".")[-1]
        if _ext == "safetensors":
            return True
        else:
            return False


    def import_on_str(
            self,
            desired_function_or_class,
            module_name = ""
            ):
        if not module_name:
            import_object = __import__(desired_function_or_class)
        else:
            import_object = getattr(__import__(module_name), desired_function_or_class)
        return import_object


    def max_temper(
            self,
            search_word,
            search_list
            ):
        return difflib.get_close_matches(search_word, search_list,cutoff=0, n=1)


    def sort_list_obj(
            self,
            list_obj,
            need_txt
            ):
        sorted_list=[]
        for module_obj in list_obj:
            if need_txt.lower() in module_obj.lower():
                sorted_list.append(module_obj)
        return sorted_list


    def sort_by_version(self,sorted_list) -> list:
        """
        Returns:
        Sorted by version in order of newest to oldest
        """
        return natsorted(sorted_list,reverse = True)
    


