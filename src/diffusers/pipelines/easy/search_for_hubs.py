import os
import re
import requests
from tqdm.auto import tqdm
from dataclasses import (
    dataclass,
    asdict
    )
from typing import Union

from ...utils import logging
from ...utils.import_utils import is_natsort_available
from ..pipeline_utils import DiffusionPipeline
from ...loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    is_valid_url
    )

from huggingface_hub import (
    hf_api,
    hf_hub_download,
    login
    )


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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_natsort_available():
    from natsort import natsorted

def sort_by_version(sorted_list) -> list:
    """
    Sorts a list by version in order of newest to oldest.
    Args:
        sorted_list (list): The list to sort.

    Returns:
        list: The sorted list.
    """
    return natsorted(sorted_list, reverse=True) if is_natsort_available() else sorted(sorted_list, reverse=True)




@dataclass
class RepoStatus:
    """
    Data class for storing repository status information.

    Attributes:
        repo_name (str): The name of the repository.
        repo_id (str): The ID of the repository.
        version_id (str): The version ID of the repository.
    """
    repo_name: str = ""
    repo_id: str = ""
    version_id: str = ""


@dataclass
class ModelStatus:
    """
    Data class for storing model status information.

    Attributes:
        search_word (str): The search word used to find the model.
        download_url (str): The URL to download the model.
        filename (str): The name of the model file.
        file_id (str): The ID of the model file.
        fp (str): Floating-point precision formats.
        local (bool): Whether the model is stored locally.
        single_file (bool): Whether the model is a single file.
    """
    search_word: str = ""
    download_url: str = ""
    filename: str = ""
    file_id: str = ""
    fp: str = ""
    local: bool = False
    single_file: bool = False


@dataclass
class SearchPipelineOutput:
    """
    Data class for storing model data.

    Attributes:
        model_path (str): The path to the model.
        load_type (str): The type of loading method used for the model.
        repo_status (RepoStatus): The status of the repository.
        model_status (ModelStatus): The status of the model.
    """
    model_path: str = ""
    load_type: str = ""  # "" or "from_single_file" or "from_pretrained"
    repo_status: RepoStatus = RepoStatus()
    model_status: ModelStatus = ModelStatus()



class HFSearchPipeline:
    """
    Huggingface class is used to search and download models from Huggingface.

    """
    model_info = {
        "model_path" : "",
        "load_type" : "",
        "repo_status":{
            "repo_name":"",
            "repo_id":"",
            "version_id":""
            },
        "model_status":{
            "search_word" : "",
            "download_url": "",
            "filename":"",
            "file_id": "",
            "fp": "",
            "local" : False,
            "single_file" : False
            },
        }
    
    def __init__(self):
        super().__init__()
        
        
    def search_for_hf(
            self,
            search_word,
            auto,
            model_format,
            model_type,
            download,
            include_civitai=True,
            include_params=False
            ):
        
        model_path = ""
        model_name = self.model_name_search(
            model_name=search_word,
            auto_set=auto,
            model_format=model_format,
            include_civitai=include_civitai
            )
        if not model_name is None:
            file_path = self.file_name_set(
                model_select=model_name,
                auto=auto,
                model_format=model_format,
                model_type=model_type
                )
            if file_path == "DiffusersFormat":
                if download:
                    model_path = self.run_hf_download(
                        model_name,
                        branch=self.branch
                        )
                else:
                    model_path = model_name
                self.model_info["model_path"] = model_path
                self.model_info["model_status"]["single_file"] = False
                self.model_info["load_type"] = "from_pretrained"

            else:
                hf_model_path = f"https://huggingface.co/{model_name}/blob/{self.branch}/{file_path}"
                if download:
                    model_path = self.run_hf_download(hf_model_path)
                else:
                    model_path = hf_model_path
                self.model_info["model_status"]["single_file"] = True
                self.model_info["load_type"] = "from_single_file"
                self.model_info["model_status"]["filename"] = file_path

            if include_params:
                return SearchPipelineOutput(
                    model_path=self.model_info["model_path"],
                    load_type=self.model_info["load_type"],
                    repo_status=RepoStatus(**self.model_info["repo_status"]),
                    model_status=ModelStatus(**self.model_info["model_status"])
                    )
            else:
                return model_path
        else:
            return None


    def repo_name_or_path(self, model_name_or_path):
        """
        Returns the repository name or path.

        Args:
            model_name_or_path (str): Model name or path.

        Returns:
            tuple: Repository ID and weights name.
        """
        pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
        weights_name = None
        repo_id = None
        for prefix in VALID_URL_PREFIXES:
            model_name_or_path = model_name_or_path.replace(prefix, "")
        match = re.match(pattern, model_name_or_path)
        if not match:
            return repo_id, weights_name
        repo_id = f"{match.group(1)}/{match.group(2)}"
        weights_name = match.group(3)
        return repo_id, weights_name


    def hf_login(self, token=None):
        """
        Logs in to Huggingface.

        Args:
            token (str): Huggingface token.
        """
        if token:
            login(token)


    def _hf_repo_download(self, path, branch="main"):
        """
        Downloads the repository.

        Args:
            path (str): Repository path.
            branch (str): Branch name.

        Returns:
            str: Path to the downloaded repository.
        """
        return DiffusionPipeline.download(
            pretrained_model_name=path,
            revision=branch,
            force_download=self.force_download,
        )


    def run_hf_download(self, url_or_path, branch="main"):
        """
        Download model from Huggingface.

        Args:
            url_or_path (str): URL or path of the model.
            branch (str): Branch name.

        Returns:
            str: Path to the downloaded model.
        """
        model_file_path = ""
        if any(url_or_path.startswith(checked) for checked in VALID_URL_PREFIXES):
            if not is_valid_url(url_or_path):
                raise requests.HTTPError(f"Invalid URL: {url_or_path}")
            hf_path, file_name = self.repo_name_or_path(url_or_path)
            logger.debug(f"url_or_path:{url_or_path}")
            logger.debug(f"hf_path: {hf_path} \nfile_name: {file_name}")
            if hf_path and file_name:
                model_file_path = hf_hub_download(
                    repo_id=hf_path,
                    filename=file_name,
                    force_download=self.force_download,
                )
            elif hf_path and (not file_name):
                if self.diffusers_model_check(hf_path):
                    model_file_path = self._hf_repo_download(
                        url_or_path, branch=branch
                    )
                else:
                    raise requests.HTTPError("Invalid hf_path")
            else:
                raise TypeError("Invalid path_or_url")
        elif self.diffusers_model_check(url_or_path):
            logger.debug(f"url_or_path: {url_or_path}")
            model_file_path = self._hf_repo_download(url_or_path, branch=branch)
        else:
            raise TypeError(f"Invalid path_or_url: {url_or_path}")
        return model_file_path


    def model_safe_check(self, model_list) -> str:
        """
        Checks if the model is safe.

        Args:
            model_list (list): List of models.

        Returns:
            str: Safe model.
        """
        if len(model_list) > 1:
            for check_model in model_list:
                if bool(re.search(r"(?i)[-_](sfw|safe)", check_model)):
                    return check_model
        return model_list[0]


    def list_safe_sort(self, model_list) -> list:
        """
        Sorts the model list by safety.

        Args:
            model_list (list): List of models.

        Returns:
            list: Sorted model list.
        """
        for check_model in model_list:
            if bool(re.search(r"(?i)[-_](sfw|safe)", check_model)):
                model_list.remove(check_model)
                model_list.insert(0, check_model)
                break
        return model_list


    def diffusers_model_check(self, checked_model: str, branch="main") -> bool:
        """
        Checks if the model is a diffuser model.

        Args:
            checked_model (str): Model to check.
            branch (str): Branch name.

        Returns:
            bool: True if the model is a diffuser model, False otherwise.
        """
        index_url = f"https://huggingface.co/{checked_model}/blob/{branch}/model_index.json"
        return is_valid_url(index_url)


    def hf_model_check(self, path) -> bool:
        """
        Checks if the model exists on Huggingface.

        Args:
            path (str): Model path.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        if not any(path.startswith(prefix) for prefix in VALID_URL_PREFIXES):
            path = f"https://huggingface.co/{path}"
        return is_valid_url(path)


    def model_data_get(self, path: str, model_info=None) -> dict:
        """
        Retrieves model data.

        Args:
            path (str): Model path.
            model_info (dict): Model information.

        Returns:
            dict: Model data.
        """
        data = model_info or self.hf_model_info(path)
        file_value_list = []
        df_model = False
        try:
            siblings = data["siblings"]
        except KeyError:
            return {}

        for item in siblings:
            file_path = item["rfilename"]
            if "model_index.json" == file_path and (not self.single_file_only):
                df_model = True
            elif (
                any(file_path.endswith(ext) for ext in EXTENSION)
                and not any(file_path.endswith(ex) for ex in CONFIG_FILE_LIST)
            ):
                file_value_list.append(file_path)
        return {
            "model_info": data,
            "file_list": file_value_list,
            "diffusers_model_exists": df_model,
            "security_risk": self.hf_security_check(data),
        }


    def hf_model_search(self, model_path, limit_num) -> list:
        """
        Searches for models on Huggingface.

        Args:
            model_path (str): Model path.
            limit_num (int): Limit number.

        Returns:
            list: List of models.
        """
        params = {
            "search": model_path,
            "sort": "likes",
            "direction": -1,
            "limit": limit_num,
            "fetch_config": True,
            "full": True,
        }
        return [asdict(value) for value in list(hf_api.list_models(**params))]


    def old_hf_model_search(self, model_path, limit_num):
        """
        Old method for searching models on Huggingface.

        Args:
            model_path (str): Model path.
            limit_num (int): Limit number.

        Returns:
            list: List of models.
        """
        url = f"https://huggingface.co/api/models"
        params = {
            "search": model_path,
            "sort": "likes",
            "direction": -1,
            "limit": limit_num,
        }
        return requests.get(url, params=params).json()


    def hf_model_info(self, model_name) -> dict:
        """
        Retrieves model information from Huggingface.

        Args:
            model_name (str): Model name.

        Returns:
            dict: Model information.
        """
        hf_info = hf_api.model_info(
            repo_id=model_name, files_metadata=True, securityStatus=True
        )
        model_dict = asdict(hf_info)
        if "securityStatus" not in model_dict.keys():
            model_dict["securityStatus"] = hf_info.__dict__["securityStatus"]
        return model_dict


    def old_hf_model_info(self, model_select) -> dict:
        """
        Old method for retrieving model information from Huggingface.

        Args:
            model_select (str): Model selection.

        Returns:
            dict: Model information.
        """
        url = f"https://huggingface.co/api/models/{model_select}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise requests.HTTPError("A hugface login or token is required")
        data = response.json()
        return data


    def hf_security_check(self, check_dict) -> int:
        """
        Check model security.

        Args:
            check_dict (dict): Model information.

        Returns:
            int: 0 for models that passed the scan, 1 for models not scanned or in error, 2 if there is a security risk.
        """
        try:
            status = check_dict["securityStatus"]
            if status["hasUnsafeFile"]:
                return 2
            elif not status["scansDone"]:
                return 1
            else:
                return 0
        except KeyError:
            return 2


    def check_if_file_exists(self, hf_repo_info):
        """
        Checks if the file exists in the Huggingface repository.

        Args:
            hf_repo_info (dict): Huggingface repository information.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            return any(
                item["rfilename"].endswith(ext)
                for item in hf_repo_info["siblings"]
                for ext in EXTENSION
            )
        except KeyError:
            return False


    def hf_models(self, model_name, limit) -> list:
        """
        Retrieve models from Huggingface.

        Args:
            model_name (str): Model name.
            limit (int): Limit number.

        Returns:
            list: List of models.
        """
        exclude_tag = ["audio-to-audio"]
        data = self.hf_model_search(model_name, limit)
        model_settings_list = []
        for item in data:
            model_id = item["id"]
            like = item["likes"]
            private_value = item["private"]
            tag_value = item["tags"]
            file_list = self.get_hf_files(item)
            diffusers_model_exists = (
                "model_index.json" in file_list and (not self.single_file_only)
            )
            if (
                all(tag not in tag_value for tag in exclude_tag)
                and (not private_value)
                and (file_list or diffusers_model_exists)
            ):
                model_dict = {
                    "model_id": model_id,
                    "like": like,
                    "model_info": item,
                    "file_list": file_list,
                    "diffusers_model_exists": diffusers_model_exists,
                    "security_risk": 1,
                }
                model_settings_list.append(model_dict)
        if not model_settings_list:
            print("No models matching your criteria were found on huggingface.")
        return model_settings_list


    def find_max_like(self, model_dict_list: list):
        """
        Finds the dictionary with the highest "like" value in a list of dictionaries.

        Args:
            model_dict_list (list): List of dictionaries.

        Returns:
            dict: The dictionary with the highest "like" value, or the first dictionary if none have "like".
        """
        max_like = 0
        max_like_dict = {}
        for model_dict in model_dict_list:
            if model_dict["like"] > max_like:
                max_like = model_dict["like"]
                max_like_dict = model_dict
        return max_like_dict["model_id"] or model_dict_list[0]["model_id"]


    def sort_by_likes(self, model_dict_list: list):
        """
        Sorts the model list by likes.

        Args:
            model_dict_list (list): List of model dictionaries.

        Returns:
            list: Sorted model list.
        """
        return sorted(model_dict_list, key=lambda x: x.get("like", 0), reverse=True)


    def get_hf_files(self, check_data) -> list:
        """
        Retrieves files from Huggingface.

        Args:
            check_data (dict): Model information.

        Returns:
            list: List of files.
        """
        check_file_value = []
        if check_data:
            siblings = check_data["siblings"]
            for item in siblings:
                fi_path = item["rfilename"]
                if (
                    any(fi_path.endswith(ext) for ext in EXTENSION)
                    and (not any(fi_path.endswith(ex) for ex in CONFIG_FILE_LIST))
                ):
                    check_file_value.append(fi_path)
        return check_file_value


    def model_name_search(
        self,
        model_name: str,
        auto_set: bool,
        model_format: str = "single_file",
        Recursive_execution: bool = False,
        extra_limit=None,
        include_civitai=False,
    ):
        """
        Search for model name on Huggingface.

        Args:
            model_name (str): Model name.
            auto_set (bool): Auto set flag.
            model_format (str): Model format.
            Recursive_execution (bool): Recursive execution flag.
            extra_limit (int): Extra limit.
            include_civitai (bool): Include Civitai flag.

        Returns:
            str: Model name.
        """
        if Recursive_execution:
            limit = 1000
        else:
            if extra_limit:
                limit = extra_limit
            else:
                limit = 15

        original_repo_model_list = self.hf_models(model_name=model_name, limit=limit)

        repo_model_list = [
            model for model in original_repo_model_list
        ]

        if not auto_set:
            print("\033[34mThe following model paths were found\033[0m")
            if include_civitai:
                print("\033[34m0.Search civitai\033[0m")
            for i, (model_dict) in enumerate(repo_model_list, 1):
                _hf_model_id = model_dict["model_id"]
                _hf_model_like = model_dict["like"]
                print(
                    f"\033[34m{i}. model path: {_hf_model_id}, evaluation: {_hf_model_like}\033[0m"
                )

            if Recursive_execution:
                print("\033[34m16.Other than above\033[0m")

            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid。\033[0m")
                    continue
                if choice == 0 and include_civitai:
                    return None
                elif (not Recursive_execution) and choice == len(repo_model_list) + 1:
                    return self.model_name_search(
                        model_name=model_name,
                        auto_set=auto_set,
                        model_format=model_format,
                        Recursive_execution=True,
                        extra_limit=extra_limit,
                    )
                elif 1 <= choice <= len(repo_model_list):
                    choice_path_dict = repo_model_list[choice - 1]
                    choice_path = choice_path_dict["model_id"]

                    security_risk = self.model_data_get(path=choice_path)["security_risk"]
                    if security_risk == 2:
                        print("\033[31mThis model has a security problem。\033[0m")
                        continue
                    else:
                        if security_risk == 1:
                            logger.warning(
                                "Warning: The specified model has not been security scanned"
                            )
                        break
                else:
                    print(
                        f"\033[34mPlease enter the numbers 1~{len(repo_model_list)}\033[0m"
                    )
        else:
            if repo_model_list:
                for check_dict in self.sort_by_likes(repo_model_list):
                    repo_info = check_dict["model_info"]
                    check_repo = check_dict["model_id"]

                    if not self.model_data_get(path=check_repo)["security_risk"] == 0:
                        continue

                    if (
                        (model_format == "diffusers" and self.diffusers_model_check(check_repo))
                        or (model_format == "single_file" and self.get_hf_files(repo_info))
                        or (
                            model_format == "all"
                            and (
                                self.diffusers_model_check(check_repo)
                                or self.get_hf_files(repo_info)
                            )
                        )
                    ):
                        choice_path = check_repo
                        break

                else:
                    if not Recursive_execution:
                        return self.model_name_search(
                            model_name=model_name,
                            auto_set=auto_set,
                            model_format=model_format,
                            Recursive_execution=True,
                            extra_limit=extra_limit,
                        )
                    else:
                        logger.warning(
                            "No models in diffusers format were found."
                        )

        return None


    def file_name_set_sub(self, model_select, file_value):
        """
        Sets the file name.

        Args:
            model_select (str): Model selection.
            file_value (list): List of file values.

        Returns:
            str: File name.
        """
        check_key = f"{model_select}_select"
        if not file_value:
            if not self.diffuser_model:
                print("\033[31mNo candidates found at huggingface\033[0m")
                res = input("Searching for civitai?: ")
                if res.lower() in ["y", "yes"]:
                    return None
                else:
                    raise ValueError(
                        "No available files were found in the specified repository"
                    )
            else:
                print("\033[34mOnly models in Diffusers format found\033[0m")
                while True:
                    result = input("Do you want to use it?[y/n]: ")
                    if result.lower() in ["y", "yes"]:
                        return "DiffusersFormat"
                    elif result.lower() in ["n", "no"]:
                        sec_result = input("Searching for civitai?[y/n]: ")
                        if sec_result.lower() in ["y", "yes"]:
                            return None
                        elif sec_result.lower() in ["n", "no"]:
                            raise ValueError(
                                "Processing was stopped because no corresponding model was found."
                            )
                    else:
                        print("\033[34mPlease enter only [y,n]\033[0m")

        file_value = self.list_safe_sort(file_value)
        if len(file_value) >= self.num_prints:
            start_number = "1"
            if self.diffuser_model:
                start_number = "0"
                print("\033[34m0.Use Diffusers format model\033[0m")
            for i in range(self.num_prints):
                print(f"\033[34m{i+1}.File name: {file_value[i]}\033[0m")
            print(
                f"\033[34m{self.num_prints+1}.Other than the files listed above (all candidates will be displayed)\033[0m\n"
            )
            while True:
                choice = input(f"select the file you want to use({start_number}~21): ")
                try:
                    choice = int(choice)
                except ValueError:
                    print("\033[33mOnly natural numbers are valid\033[0m")
                    continue
                if self.diffuser_model and choice == 0:
                    return "DiffusersFormat"

                elif choice == (self.num_prints + 1):
                    break
                elif 1 <= choice <= self.num_prints:
                    choice_path = file_value[choice - 1]
                    return choice_path
                else:
                    print(
                        f"\033[33mPlease enter numbers from 1~{self.num_prints}\033[0m"
                    )
            print("\n\n")
        start_number = "1"
        if self.diffuser_model:
            start_number = "0"
            print("\033[34m0.Use Diffusers format model\033[0m")
        for i, file_name in enumerate(file_value, 1):
            print(f"\033[34m{i}.File name: {file_name}\033[0m")
        while True:
            choice = input(
                f"Select the file you want to use({start_number}~{len(file_value)}): "
            )
            try:
                choice = int(choice)
            except ValueError:
                print("\033[33mOnly natural numbers are valid\033[0m")
            else:
                if self.diffuser_model and choice == 0:
                    return "DiffusersFormat"

                if 1 <= choice <= len(file_value):
                    choice_path = file_value[choice - 1]
                    return choice_path
                else:
                    print(
                        f"\033[33mPlease enter numbers from 1~{len(file_value)}\033[0m"
                    )


    def file_name_set(
        self, model_select, auto, model_format, model_type="Checkpoint"
    ):
        """
        Sets the file name.

        Args:
            model_select (str): Model selection.
            auto (bool): Auto flag.
            model_format (str): Model format.
            model_type (str): Model type.

        Returns:
            str: File name.
        """
        if self.diffusers_model_check(model_select) and model_type == "Checkpoint":
            self.diffuser_model = True
        else:
            self.diffuser_model = False

        if model_format == "single_file":
            skip_difusers = True
        else:
            skip_difusers = False

        data = self.hf_model_info(model_select)
        choice_path = ""
        file_value = []
        if data:
            file_value = self.get_hf_files(check_data=data)
        else:
            raise ValueError("No available file was found. Please check the name.")
        if file_value:
            file_value = self.sort_by_version(file_value)
            if not auto:
                print("\033[34mThe following model files were found\033[0m")
                choice_path = self.file_name_set_sub(model_select, file_value)
            else:
                if self.diffuser_model and (not skip_difusers):
                    choice_path = "DiffusersFormat"
                else:
                    choice_path = self.model_safe_check(file_value)

        elif self.diffuser_model:
            choice_path = "DiffusersFormat"
        else:
            raise FileNotFoundError(
                "No available files found in the specified repository"
            )
        return choice_path
    



class CivitaiSearchPipeline:
    """
    The Civitai class is used to search and download models from Civitai.

    Attributes:
        base_civitai_dir (str): Base directory for Civitai.
        max_number_of_choices (int): Maximum number of choices.
        chunk_size (int): Chunk size.

    Methods:
        for_civitai(search_word, auto, model_type, download, civitai_token, skip_error, include_hugface):
            Downloads a model from Civitai.
        civitai_security_check(value): Performs a security check.
        requests_civitai(query, auto, model_type, civitai_token, include_hugface): Retrieves models from Civitai.
        repo_select_civitai(state, auto, recursive, include_hugface): Selects a repository from Civitai.
        download_model(url, save_path, civitai_token): Downloads a model.
        version_select_civitai(state, auto, recursive): Selects a model version from Civitai.
        file_select_civitai(state_list, auto, recursive): Selects a file to download.
        civitai_save_path(): Sets the save path.
    """

    base_civitai_dir = "/root/.cache/Civitai"
    max_number_of_choices: int = 15
    chunk_size: int = 8192

    def __init__(self):
        super().__init__()


    def __call__(
        self,
        **keywords
    ):
        return self.civitai_model_set(**keywords)


    def civitai_model_set(
        self,
        search_word,
        **kwargs
    ) -> SearchPipelineOutput:
        """
        Downloads a model from Civitai.

        Parameters:
        - search_word (str): Search query string.
        - auto (bool): Auto-select flag.
        - model_type (str): Type of model to search for.
        - download (bool): Whether to download the model.
        - include_params (bool): Whether to include parameters in the returned data.

        Returns:
        - SearchPipelineOutput
        """
        auto = kwargs.pop("auto", True)
        model_type = kwargs.pop("model_type", "Checkpoint")
        download = kwargs.pop("download", False)
        civitai_token = kwargs.pop("civitai_token", None)
        include_params = kwargs.pop("include_params", False)
        include_hugface = kwargs.pop("include_hugface",True)
        skip_error = kwargs.pop("skip_error", True)

        model_info = {
            "model_path" : "",
            "load_type" : "",
            "repo_status":{
                "repo_name":"",
                "repo_id":"",
                "version_id":""
                },
            "model_status":{
                "search_word" : "",
                "download_url": "",
                "filename":"",
                "file_id": "",
                "fp": "",
                "local" : False,
                "single_file" : False
                },
            }

        state = self.requests_civitai(
            query=search_word,
            auto=auto,
            model_type=model_type,
            civitai_token=civitai_token,
            include_hugface=include_hugface,
        )
        if not state:
            if skip_error:
                return None
            else:
                raise ValueError("No models were found in civitai.")

        dict_of_civitai_repo = self.repo_select_civitai(
            state=state, auto=auto, include_hugface=include_hugface
        )

        if not dict_of_civitai_repo:
            return None

        files_list = self.version_select_civitai(state=dict_of_civitai_repo, auto=auto)

        file_status_dict = self.file_select_civitai(state_list=files_list, auto=auto)
        
        model_download_url = file_status_dict["download_url"]
        model_info["repo_status"]["repo_name"] = dict_of_civitai_repo["repo_name"]
        model_info["repo_status"]["repo_id"] = dict_of_civitai_repo["repo_id"]
        model_info["repo_status"]["version_id"] = files_list["id"]
        model_info["model_status"]["download_url"] = model_download_url
        model_info["model_status"]["filename"] = file_status_dict["filename"]
        model_info["model_status"]["file_id"] = file_status_dict["file_id"]
        model_info["model_status"]["fp"] = file_status_dict["fp"]
        model_info["model_status"]["file_format"] = file_status_dict["file_format"]
        model_info["model_status"]["filename"] = file_status_dict["filename"]
        model_info["model_status"]["single_file"] = True
        if download:
            model_save_path = self.civitai_save_path()
            model_info["model_path"] = model_save_path
            model_info["load_type"] = "from_single_file"
            self.download_model(
                url=model_download_url,
                save_path=model_save_path,
                civitai_token=civitai_token,
            )
        else:
            model_info["model_path"] = model_info["model_status"]["download_url"]
            model_info["load_type"] = ""
    
        if not include_params:
            return model_info["model_path"]
        else:
            return SearchPipelineOutput(
                model_path=model_info["model_path"],
                load_type=model_info["load_type"],
                repo_status=RepoStatus(**model_info["repo_status"]),
                model_status=ModelStatus(**model_info["model_status"])
            )


    def civitai_security_check(
        self,
        value
    ) -> int:
        """
        Performs a security check.

        Parameters:
        - value: Value to check.

        Returns:
        - int: Security risk level.
        """
        try:
            pickleScan = value["pickleScanResult"]
            virusScan = value["virusScanResult"]
        except KeyError:
            return 1
        check_list = [pickleScan, virusScan]
        if all(status == "Success" for status in check_list):
            return 0
        elif "Danger" in check_list:
            return 2
        else:
            return 1


    def requests_civitai(
        self,
        query,
        auto,
        model_type,
        civitai_token=None,
        include_hugface=True
    ):
        """
        Retrieves models from Civitai.

        Parameters:
        - query: Search query string.
        - model_type: Type of model to search for.

        Returns:
        - list: List of model information.
        """
        state = []
        model_ver_list = []
        version_dict = {}

        params = {"query": query, "types": model_type, "sort": "Most Downloaded"}

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

        try:
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
                raise ValueError("Invalid JSON response")

        for item in data["items"]:
            for model_ver in item["modelVersions"]:
                files_list = []
                for model_value in model_ver["files"]:
                    security_risk = self.civitai_security_check(model_value)
                    if (
                        any(
                            check_word in model_value
                            for check_word in ["downloadUrl", "name"]
                        )
                        and not security_risk
                    ):
                        file_status = {
                            "filename": model_value["name"],
                            "file_id": model_value["id"],
                            "fp": model_value["metadata"]["fp"],
                            "file_format": model_value["metadata"]["format"],
                            "download_url": model_value["downloadUrl"],
                        }
                        files_list.append(file_status)

                version_dict = {
                    "id": model_ver["id"],
                    "name": model_ver["name"],
                    "downloadCount": model_ver["stats"]["downloadCount"],
                    "files": files_list,
                }

                if files_list:
                    model_ver_list.append(version_dict)

            if all(
                check_txt in item.keys()
                for check_txt in ["name", "stats", "creator"]
            ):
                state_dict = {
                    "repo_name": item["name"],
                    "repo_id": item["id"],
                    "favoriteCount": item["stats"]["favoriteCount"],
                    "downloadCount": item["stats"]["downloadCount"],
                    "CreatorName": item["creator"]["username"],
                    "version_list": model_ver_list,
                }

                if model_ver_list:
                    yield state_dict


    def repo_select_civitai(
        self,
        state: list,
        auto: bool,
        recursive: bool = True,
        include_hugface: bool = True
    ):
        """
        Selects a repository from Civitai.

        Parameters:
        - state (list): List of repository information.
        - auto (bool): Auto-select flag.
        - recursive (bool): Recursive flag.

        Returns:
        - dict: Selected repository information.
        """
        if not state:
            logger.warning("No models were found in civitai.")
            return {}

        elif auto:
            repo_dict = max(state, key=lambda x: x["downloadCount"])
            return repo_dict
        else:
            sorted_list = sorted(state, key=lambda x: x["downloadCount"], reverse=True)
            if recursive and self.max_number_of_choices < len(sorted_list):
                Limit_choice = True
            else:
                Limit_choice = False

            if recursive:
                print("\n\n\033[34mThe following repo paths were found\033[0m")
            else:
                print("\n\n\n")

            max_number = (
                min(self.max_number_of_choices, len(sorted_list))
                if recursive
                else len(sorted_list)
            )
            if include_hugface:
                print(f"\033[34m0. Search for huggingface\033[0m")
            for number, states_dict in enumerate(sorted_list[:max_number]):
                print(
                    f"\033[34m{number + 1}. Repo_id: {states_dict['CreatorName']} / {states_dict['repo_name']}, download: {states_dict['downloadCount']}\033[0m"
                )

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")

            while True:
                try:
                    choice = int(input(f"choice repo [1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid。\033[0m")
                    continue

                if Limit_choice and choice == max_number:
                    return self.repo_select_civitai(
                        state=state, auto=auto, recursive=False
                    )
                elif choice == 0 and include_hugface:
                    return {}
                elif 1 <= choice <= max_number:
                    repo_dict = sorted_list[choice - 1]
                    return repo_dict
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[0m")


    def download_model(
        self,
        url,
        save_path,
        civitai_token=None
    ):
        """
        Downloads a model.

        Parameters:
        - url (str): Download URL.
        - save_path (str): Save path.
        - civitai_token (str): Civitai token.
        """
        if not is_valid_url(url):
            raise requests.HTTPError("URL is invalid.")

        headers = {}
        if civitai_token:
            headers["Authorization"] = f"Bearer {civitai_token}"

        response = requests.get(url, stream=True, headers=headers)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Invalid URL: {url}, {response.status_code}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with tqdm.wrapattr(
            open(save_path, "wb"),
            "write",
            miniters=1,
            desc=os.path.basename(save_path),
            total=int(response.headers.get("content-length", 0)),
        ) as fout:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                fout.write(chunk)
        logger.info(f"Downloaded file saved to {save_path}")


    def version_select_civitai(
        self,
        state,
        auto,
        recursive: bool = True
    ):
        """
        Selects a model version from Civitai.

        Parameters:
        - state: Model information state.
        - auto: Auto-select flag.
        - recursive (bool): Recursive flag.

        Returns:
        - dict: Selected model information.
        """
        if not state:
            raise ValueError("state is empty")

        ver_list = sorted(state["version_list"], key=lambda x: x["downloadCount"], reverse=True)

        if recursive and self.max_number_of_choices < len(ver_list):
            Limit_choice = True
        else:
            Limit_choice = False

        if auto:
            result = max(ver_list, key=lambda x: x["downloadCount"])
            version_files_list = self.sort_by_version(result["files"])
            self.model_info["repo_status"]["version_id"] = result["id"]
            return version_files_list
        else:
            if recursive:
                print("\n\n\033[34mThe following model paths were found\033[0m")
            else:
                print("\n\n\n")

            if len(ver_list) == 1:
                return ver_list

            max_number = (
                min(self.max_number_of_choices, len(ver_list))
                if recursive
                else len(ver_list)
            )

            for number_, state_dict_ in enumerate(ver_list[:max_number]):
                print(
                    f"\033[34m{number_ + 1}. model_version: {state_dict_['name']}, download: {state_dict_['downloadCount']}\033[0m"
                )

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")

            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid。\033[0m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.version_select_civitai(
                        state=state, auto=auto, recursive=False
                    )
                elif 1 <= choice <= max_number:
                    return_dict = ver_list[choice - 1]
                    return return_dict["files"]
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[0m")


    def file_select_civitai(
        self,
        state_list,
        auto,
        recursive: bool = True
    ):
        """
        Selects a file to download.

        Parameters:
        - state_list: List of file information.
        - auto: Auto-select flag.

        Returns:
        - dict: Selected file information.
        """
        if recursive and self.max_number_of_choices < len(state_list):
            Limit_choice = True
        else:
            Limit_choice = False

        if len(state_list) > 1 and (not auto):
            max_number = (
                len(state_list)
                if not recursive
                else min(self.max_number_of_choices, len(state_list))
            )
            for number, states_dict in enumerate(state_list[:max_number]):
                print(f"\033[34m{number + 1}. File_name: {states_dict['filename']}")

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")

            while True:
                try:
                    choice = int(input(f"Select the file to download[1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid。\033[0m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.file_select_civitai(
                        state_list=state_list, auto=auto, recursive=False
                    )
                elif 1 <= choice <= len(state_list):
                    file_dict = state_list[choice - 1]
                    return file_dict
                else:
                    print(f"\033[33mPlease enter the numbers 1~{len(state_list)}\033[0m")
        else:
            file_dict = state_list[0]
            return file_dict


    def civitai_save_path(
        self
    ) -> os.PathLike:
        """
        Sets the save path.

        Returns:
        - str: Save path.
        """
        repo_level_dir = str(self.model_info["repo_status"]["repo_id"])
        file_version_dir = str(self.model_info["repo_status"]["version_id"])
        save_file_name = str(self.model_info["model_status"]["filename"])
        save_path = os.path.join(
            self.base_civitai_dir, repo_level_dir, file_version_dir, save_file_name
        )
        return save_path




class SearchPipeline(
    HFSearchPipeline,
    CivitaiSearchPipeline
    ):
    def __init__(self):
        super().__init__()
    

    def search_for_hubs(
            self,
            search_word: str,
            **kwargs
            ) -> Union[str, SearchPipelineOutput]:
        """Search and retrieve model information from various sources.

        Args:
            search_word: The search term to find the model.
            auto: Whether to automatically select the best match.
            download: Whether to download the model locally.
            model_type: Type of model (e.g., "Checkpoint", "LORA").
            model_format: Format of the model ("single_file", "diffusers", "all").
            branch: The repository branch to search in.
            priority_hub: Which model hub to prioritize ("huggingface" or "civitai").
            local_file_only: Whether to search only in local files.
            include_params: Whether to include additional model parameters in output.
            hf_token: HuggingFace API token for authentication.
            civitai_token: Civitai API token for authentication.            

        Returns:
            Either a string path to the model or a SearchPipelineOutput object with
            full model information if include_params is True.

        Raises:
            ValueError: If the model cannot be found or accessed.
        """
        auto = kwargs.pop("auto", True)
        download = kwargs.pop("download", False)
        model_type = kwargs.pop("model_type", "single_file")
        model_format = kwargs.pop("model_format", "Checkpoint")
        branch = kwargs.pop("branch", "main")
        priority_hub = kwargs.pop("priority_hub", "huggingface")
        include_params = kwargs.pop("include_params", False)
        local_file_only = kwargs.pop("local_search_only", False)
        civitai_token = kwargs.pop("civitai_token", None)

        if "hf_token" in kwargs:
            hf_token = kwargs.pop("hf_token", None)
            login(token=hf_token)        
        
        self.single_file_only = True if "single_file" == model_format else False

        self.model_info["model_status"]["search_word"] = search_word
        self.model_info["model_status"]["local"] = True if download or local_file_only else False

        result = self.model_set(
            model_select=search_word,
            auto=auto,
            download=download,
            model_format=model_format,
            model_type=model_type,
            branch=branch,
            priority_hub=priority_hub,
            local_file_only=local_file_only,
            civitai_token=civitai_token,
            include_params=include_params
        )
                
        if not include_params:
            return result
        else:
            return SearchPipelineOutput(
                model_path=self.model_info["model_path"],
                load_type=self.model_info["load_type"],
                repo_status=RepoStatus(**self.model_info["repo_status"]),
                model_status=ModelStatus(**self.model_info["model_status"])
            )

    def File_search(
        self,
        search_word,
        auto=True
    ):
        """
        Search for files based on the provided search word.

        Args:
            search_word (str): The search word for the file.
            auto (bool, optional): Whether to enable auto mode. Defaults to True.

        Yields:
            str: The path of the found file.
        """
        closest_match = None
        closest_distance = float('inf')
        for root, dirs, files in os.walk("/"):
            for file in files:
                if any(file.endswith(ext) for ext in EXTENSION):
                    path = os.path.join(root, file)
                    if not any(path.endswith(ext) for ext in CONFIG_FILE_LIST):
                        yield path
                        if auto:
                            distance = self.calculate_distance(search_word, file)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_match = path
        if auto:
            return closest_match
        else:
            return self.user_select_file(search_word)
    

    def user_select_file(self, search_word, result):
        """
        Allow user to select a file from the search results.

        Args:
            search_word (str): The search word for the file.

        Returns:
            str: The path of the selected file.

        Raises:
            FileNotFoundError: If no files are found.
        """
        search_results = list(self.File_search(search_word, auto=False))
        if not search_results:
            raise FileNotFoundError("\033[33mModel File not found\033[0m")
        
        print("\n\n\033[34mThe following model files were found\033[0m")
        for i, path in enumerate(search_results, 1):
            print(f"\033[34m{i}. {path}\033[0m")
        
        while True:
            try:
                choice = int(input(f"Select the file to use [1-{len(search_results)}]: "))
                if 1 <= choice <= len(search_results):
                    return search_results[choice - 1]
                else:
                    print(f"\033[33mPlease enter a number between 1 and {len(search_results)}\033[0m")
            except ValueError:
                print("\033[33mOnly natural numbers are valid.\033[0m")
    

    def calculate_distance(self, search_word, file_name):
        """
        Calculate the distance between the search word and the file name.

        Args:
            search_word (str): The search word.
            file_name (str): The file name.

        Returns:
            int: The distance between the search word and the file name.
        """
        return sum(1 for a, b in zip(search_word, file_name) if a != b)  


    def model_set(
        self,
        model_select,
        auto=True,
        download=False,
        model_format="single_file",
        model_type="Checkpoint",
        branch="main",
        priority_hub="hugface",
        local_file_only=False,
        civitai_token=None,
        include_params=False
    ):
        """
        Set the model based on the provided parameters.

        Args:
            model_select (str): The model selection criteria.
            auto (bool, optional): Whether to enable auto mode. Defaults to True.
            download (bool, optional): Whether to download the model. Defaults to False.
            model_format (str, optional): The format of the model. Defaults to "single_file".
            model_type (str, optional): The type of the model. Defaults to "Checkpoint".
            branch (str, optional): The branch of the model. Defaults to "main".
            priority_hub (str, optional): The priority hub. Defaults to "hugface".
            local_file_only (bool, optional): Whether to search locally only. Defaults to False.
            civitai_token (str, optional): The Civitai token. Defaults to None.
            include_params (bool, optional): Whether to include parameters in the returned data. Defaults to False.

        Returns:
            ModelData or str: The model data if include_params is True, otherwise the model path.

        Raises:
            TypeError: If the model_type or model_format is invalid.
            ValueError: If the specified repository could not be found.
            FileNotFoundError: If the model_index.json file is not found.
        """
        if not model_type in ["Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses"]:
            raise TypeError(f'Wrong argument. Valid values are "Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses". What was passed on {model_type}')
        
        if not model_format in ["all","diffusers","single_file"]:
            raise TypeError('The model_format is valid only for one of the following: "all","diffusers","single_file"')
      
        model_path = model_select
        file_path = ""
        if model_select in self.model_dict:
            model_path_to_check = self.model_dict[model_select]
            _check_url = f"https://huggingface.co/{model_path_to_check}"
            if is_valid_url(_check_url):
                model_select = model_path_to_check
                self.model_info["model_path"] = _check_url
            else:
                logger.warning(f"The following custom search keys are ignored.`{model_select} : {CUSTOM_SEARCH_KEY[model_select]}`")

        if local_file_only:
            model_path = next(self.File_search(
                search_word=model_select,
                auto=auto
            ))
            self.model_info["model_status"]["single_file"] = True
            self.model_info["model_path"] = model_path
            self.model_info["load_type"] = "from_single_file"

        elif model_select.startswith("https://huggingface.co/"):
            if not is_valid_url(model_select):
                raise ValueError("Could not load URL")
            else:
                if download:
                    model_path = self.run_hf_download(model_select)
                else:
                    model_path = model_select

                self.model_info["model_status"]["single_file"] = True
                self.model_info["model_path"] = model_path
                repo, file_name = self.repo_name_or_path(model_select)
                if file_name:
                    self.model_info["model_status"]["filename"] = file_name
                    self.model_info["model_status"]["single_file"] = True
                    self.model_info["load_type"] = "from_single_file"
                else:
                    self.model_info["model_status"]["single_file"] = False
                    self.model_info["load_type"] = "from_pretrained"


        elif model_select.startswith("https://civitai.com/"):
            model_path = self.civitai_model_set(
                search_word=model_select,
                auto=auto,
                model_type=model_type,
                download=download,
                civitai_token=civitai_token,
                skip_error=False
            )

        elif os.path.isfile(model_select):
            model_path = model_select
            self.model_info["model_path"] = model_select
            self.model_info["model_status"]["single_file"] = True
            self.model_info["load_type"] = "from_single_file"
            self.model_info["model_status"]["local"] = True

        elif os.path.isdir(model_select):
            if os.path.exists(os.path.join(model_select, self.Config_file)):
                model_path = model_select
                self.model_info["model_path"] = model_select
                self.model_info["model_status"]["single_file"] = False
                self.model_info["load_type"] = "from_pretrained"
                self.model_info["model_status"]["local"] = True
            else:
                raise FileNotFoundError(f"model_index.json not found in {model_select}")

        elif model_select.count("/") == 1:
            creator_name, repo_name = model_select.split("/")

            if auto and self.diffusers_model_check(model_select):
                if download:
                    model_path = self.run_hf_download(model_select)
                    self.model_info["model_status"]["single_file"] = False
                else:
                    model_path = model_select
                    self.model_info["model_status"]["single_file"] = False
                self.model_info["load_type"] = "from_pretrained"

            elif auto and (not self.hf_model_check(model_select)):
                raise ValueError(f'The specified repository could not be found, please try turning off "auto" (model_select:{model_select})')
            else:
                file_path=self.file_name_set(model_select,auto,model_type)
                if file_path is None:
                    raise ValueError("Model not found")
                elif file_path == "DiffusersFormat":
                    if download:
                        model_path = self.run_hf_download(model_select)
                    else:
                        model_path = model_select

                    self.model_info["model_status"]["single_file"] = False
                    self.model_info["load_type"] = "from_pretrained"
                    
                else:
                    hf_model_path = f"https://huggingface.co/{model_select}/blob/{branch}/{file_path}"
                    
                    if download:
                        model_path = self.run_hf_download(hf_model_path)
                    else:
                        model_path = hf_model_path
                    
                    self.model_info["model_status"]["filename"] = file_path
                    self.model_info["model_status"]["single_file"] = True
                    self.model_info["load_type"] = "from_single_file"
                
            self.model_info["repo_status"]["repo_name"] = repo_name
                
        else:
            if priority_hub == "huggingface":
                model_path = self.hf_model_set(
                    model_select=model_select,
                    auto=auto,
                    model_format=model_format,
                    model_type=model_type,
                    download=download,
                    include_civitai=True
                    )
                if model_path is None:
                    model_path = self.civitai_model_set(
                        search_word=model_select,
                        auto=auto,
                        model_type=model_type,
                        download=download,
                        civitai_token=civitai_token,
                        include_hugface=False
                    )
                    if not model_path:
                        raise ValueError("No models matching the criteria were found.")
                
            else:
                model_path = self.civitai_model_set(
                    search_word=model_select,
                    auto=auto,
                    model_type=model_type,
                    download=download,
                    civitai_token=civitai_token,
                    include_hugface=True
                )
                if not model_path:
                    model_path = self.hf_model_set(
                        model_select=model_select,
                        auto=auto,
                        model_format=model_format,
                        model_type=model_type,
                        download=download,
                        include_civitai=False
                        )
                    if model_path is None:
                        raise ValueError("No models matching the criteria were found.")
                
        self.model_info["model_path"] = model_path
        if include_params:
            return self.model_info
        else:
            return model_path



class ModelSearchPipeline(SearchPipeline):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def for_hubs(
        cls,
        search_word,
        **kwargs
    ):
        return cls().search_for_hubs(cls, search_word, **kwargs)
    
    @classmethod
    def for_HF(
        cls,
        search_word,
        **kwargs
    ):
        return cls().search_for_hf(cls, search_word, **kwargs)
    
    @classmethod
    def for_civitai(
        cls,
        search_word,
        **kwargs
    ):
        return cls().search_for_civitai(cls, search_word, **kwargs)