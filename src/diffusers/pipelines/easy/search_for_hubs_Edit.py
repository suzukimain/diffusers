import os
import re
import requests
from typing import Union
from tqdm.auto import tqdm
from dataclasses import asdict
from huggingface_hub import (
    hf_api,
    hf_hub_download,
    login,
)

from ...utils import logging
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import (
    SearchPipelineOutput,
    ModelStatus,
    RepoStatus,
)
from ...loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    is_valid_url,
    _extract_repo_id_and_weights_name,
)


CUSTOM_SEARCH_KEY = {
    "sd" : "stabilityai/stable-diffusion-2-1",
    }


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
    "safety_checker/model.safetensors",
    "safety_checker/model.ckpt",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.fp16.ckpt",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.ckpt",
    "unet/diffusion_pytorch_model.fp16.ckpt",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
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



def get_keyword_types(keyword):
    """
    Determine the type and loading method for a given keyword.
    
    Args:
        keyword (str): The input keyword to classify.
        
    Returns:
        dict: A dictionary containing the model format, loading method,
              and various types and extra types flags.
    """
    
    # Initialize the status dictionary with default values
    status = {
        "model_format": None,
        "loading_method": None,
        "type": {
            "search_word": False,
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
        status["model_format"] = "single_file"
        status["loading_method"] = "from_single_file"
    
    # Check if the keyword is a directory
    elif os.path.isdir(keyword):
        status["type"]["local"] = True
        status["model_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
        if not os.path.exists(os.path.join(keyword, "model_index.json")):
            status["extra_type"]["missing_model_index"] = True
    
    # Check if the keyword is a Civitai URL
    elif keyword.startswith("https://civitai.com/"):
        status["type"]["civitai_url"] = True
        status["model_format"] = "single_file"
        status["loading_method"] = None
    
    # Check if the keyword starts with any valid URL prefixes
    elif any(keyword.startswith(prefix) for prefix in VALID_URL_PREFIXES):
        repo_id, weights_name = _extract_repo_id_and_weights_name(keyword)
        if weights_name:
            status["type"]["hf_url"] = True
            status["model_format"] = "single_file"
            status["loading_method"] = "from_single_file"
        else:
            status["type"]["hf_repo"] = True
            status["model_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"
    
    # Check if the keyword matches a Hugging Face repository format
    elif re.match(r"^[^/]+/[^/]+$", keyword):
        status["type"]["hf_repo"] = True
        status["model_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
    
    # If none of the above, treat it as a search word
    else:
        status["type"]["search_word"] = True
        status["model_format"] = None
        status["loading_method"] = None
    
    return status







class HFSearchPipeline:
    """
    Search for models from Huggingface.

    Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    """
    model_info = {
        "model_path": "",
        "load_type": "",
        "repo_status": {
            "repo_name": "",
            "repo_id": "",
            "revision": ""
        },
        "model_status": {
            "search_word": "",
            "download_url": "",
            "filename": "",
            "local": False,
            "single_file": False
        },
    }

    def __init__(self):
        pass
    
    

    @staticmethod
    def create_huggingface_url(repo_id, file_name):
        """
        Create a Hugging Face URL for a given repository ID and file name.
        
        Args:
            repo_id (str): The repository ID.
            file_name (str): The file name within the repository.
        
        Returns:
            str: The complete URL to the file or repository on Hugging Face.
        """
        if file_name:
            return f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        else:
            return f"https://huggingface.co/{repo_id}"
    
    @staticmethod
    def hf_find_safest_model(models) -> str:
        """
        Sort and find the safest model.

        Args:
            models (list): A list of model names to sort and check.

        Returns:
            The name of the safest model or the first model in the list if no safe model is found.
        """
        for model in sorted(models, reverse=True):
            if bool(re.search(r"(?i)[-_](safe|sfw)", model)):
                return model
        return models[0]


    @classmethod
    def for_HF(cls, search_word, **kwargs):
        """
        Class method to search and download models from Hugging Face.
        
        Args:
            search_word (str): The search keyword for finding models.
            **kwargs: Additional keyword arguments.
        
        Returns:
            str: The path to the downloaded model or search word.
        """
        # Extract additional parameters from kwargs
        revision = kwargs.pop("revision", None)
        model_format = kwargs.pop("model_format", "single_file")
        download = kwargs.pop("download", False)
        force_download = kwargs.pop("force_download", False)
        include_params = kwargs.pop("include_params", False)
        pipeline_tag = kwargs.pop("pipeline_tag", None)
        hf_token = kwargs.pop("hf_token", None)
        skip_error = kwargs.pop("skip_error", False)

        # Get the type and loading method for the keyword
        search_word_status = get_keyword_types(search_word)

        # Handle different types of keywords
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
                model_path = None
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
                diffusers_model_exists = False
                if hf_security_info["scansDone"]:
                    for info in repo_info["siblings"]:
                        file_path = info["rfilename"]
                        if (
                            "model_index.json" == file_path
                            and model_format in ["diffusers", "all"]
                        ):
                            diffusers_model_exists = True
                            break
                        
                        elif (
                            any(file_path.endswith(ext) for ext in EXTENSION)
                            and (file_path not in CONFIG_FILE_LIST)
                            and (file_path not in exclusion)
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
                model_format=output_info["model_format"],
                repo_status=RepoStatus(
                    repo_id=repo_id,
                    repo_hash=hf_repo_info["sha"],
                    revision=revision
                ),
                model_status=ModelStatus(
                    search_word=search_word,
                    download_url=download_url,
                    filename=file_name,
                    local=download,
                )
            )
        
        else:
            return model_path    



    



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
        pass

    @staticmethod
    def civitai_find_safest_model(models) -> str:
        """
        Sort and find the safest model.

        Args:
            models (list): A list of model names to sort and check.

        Returns:
            The name of the safest model or the first model in the list if no safe model is found.
        """

        for model in sorted(models,key=lambda x: x["filename"], reverse=True):
            if bool(re.search(r"(?i)[-_](safe|sfw)", model["filename"])):
                return model
        return models[0]

    @classmethod
    def for_civitai(
        cls,
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
        model_format = kwargs.pop("model_format", "single_file")
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
                "revision":""
                },
            "model_status":{
                "search_word" : "",
                "download_url": "",
                "filename":"",
                "local" : False,
                "single_file" : False
                },
            }
        
        cls.single_file_only = True if "single_file" == model_format else False

        cls.model_info["model_status"]["search_word"] = search_word
        cls.model_info["model_status"]["local"] = True if download else False

        state = cls.requests_civitai(
            query=search_word,
            model_type=model_type,
            civitai_token=civitai_token,
        )
       
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

        for items in data["items"]:
            sorted_repos = sorted(items, key=lambda x: x["stats"]["downloadCount"], reverse=True)
            for model_versions in sorted_repos["modelVersions"]:
                files_list = []
                sorted_model_versions = sorted(model_versions, key=lambda x: x["stats"]["downloadCount"], reverse=True)
                for model_value in sorted_model_versions["files"]:

                    if all(
                        model_value["pickleScanResult"] == "Success"
                        and model_value["virusScanResult"] == "Success"
                    ):
                        file_status = {
                            "filename": model_value["name"],
                            "download_url": model_value["downloadUrl"],
                        }
                        files_list.append(file_status)
                
                if files_list:
                    model_path = cls.civitai_find_safest_model(files_list)




        dict_of_civitai_repo = cls.repo_select_civitai(
            state=state, auto=auto, include_hugface=include_hugface
        )

        if not dict_of_civitai_repo:
            return None
        
        version_data = sorted(dict_of_civitai_repo["version_list"], key=lambda x: x["downloadCount"], reverse=True)[0]["files"]

        files_list = version_data["files"]
        file_status_dict = cls.file_select_civitai(state_list=files_list, auto=auto)
        
        model_download_url = file_status_dict["download_url"]
        model_info["repo_status"]["repo_name"] = dict_of_civitai_repo["repo_name"]
        model_info["repo_status"]["repo_id"] = dict_of_civitai_repo["repo_id"]
        model_info["repo_status"]["revision"] = version_data["id"]
        model_info["model_status"]["download_url"] = model_download_url
        model_info["model_status"]["filename"] = file_status_dict["filename"]
        #model_info["model_status"]["file_format"] = file_status_dict["file_format"]
        model_info["model_status"]["single_file"] = True
        if download:
            model_save_path = cls.civitai_save_path()
            model_info["model_path"] = model_save_path
            model_info["load_type"] = "from_single_file"
            cls.download_model(
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
        model_type,
        civitai_token=None,
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

        for items in data["items"]:
            sorted_repos = sorted(items, key=lambda x: x["stats"]["downloadCount"], reverse=True)
            for model_versions in sorted_repos["modelVersions"]:
                files_list = []
                sorted_model_versions = sorted(model_versions, key=lambda x: x["stats"]["downloadCount"], reverse=True)
                for model_value in sorted_model_versions["files"]:

                    if all(
                        model_value["pickleScanResult"] == "Success"
                        and model_value["virusScanResult"] == "Success"
                    ):
                        file_status = {
                            "filename": model_value["name"],
                            "download_url": model_value["downloadUrl"],
                        }
                        files_list.append(file_status)
                
                if files_list:
                    model_path = find_safest_model(files_list)
                    
                    sorted_files_list = natural_sort(files_list)
                    version_dict = {
                        "id": model_ver["id"],
                        "name": model_ver["name"],
                        "downloadCount": model_ver["stats"]["downloadCount"],
                        "files": sorted_files_list,
                    }
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

        ver_list = sorted(state["version_list"], key=lambda x: x["downloadCount"], reverse=True)[0]

        if recursive and self.max_number_of_choices < len(ver_list):
            Limit_choice = True
        else:
            Limit_choice = False

        if auto:
            result = max(ver_list, key=lambda x: x["downloadCount"])
            #version_files_list = natural_sort(result["files"])
            #self.model_info["repo_status"]["revision"] = result["id"]
            return result
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
                    return return_dict
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
        file_version_dir = str(self.model_info["repo_status"]["revision"])
        save_file_name = str(self.model_info["model_status"]["filename"])
        save_path = os.path.join(
            self.base_civitai_dir, repo_level_dir, file_version_dir, save_file_name
        )
        return save_path




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
    ) -> Union[str, SearchPipelineOutput]:
        """Search and retrieve model information from various sources.

        Args:
            search_word: The search term to find the model.
            auto: Whether to automatically select the best match.
            download: Whether to download the model locally.
            model_type: Type of model ("single_file", "diffusers", "all").
            model_format: Format of the model (e.g., "Checkpoint", "LORA").
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
        model_type = kwargs.pop("model_type", "Checkpoint")
        model_format = kwargs.pop("model_format","single_file" )
        branch = kwargs.pop("branch", "main")
        priority_hub = kwargs.pop("priority_hub", "huggingface")
        include_params = kwargs.pop("include_params", False)
        local_file_only = kwargs.pop("local_search_only", False)
        civitai_token = kwargs.pop("civitai_token", None)

        if "hf_token" in kwargs:
            hf_token = kwargs.pop("hf_token", None)
            login(token=hf_token)        
        
        cls.single_file_only = True if "single_file" == model_format else False

        cls.model_info["model_status"]["search_word"] = search_word
        cls.model_info["model_status"]["local"] = True if download or local_file_only else False

        result = cls.model_set(
            search_word=search_word,
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
                model_path=cls.model_info["model_path"],
                load_type=cls.model_info["load_type"],
                repo_status=RepoStatus(**cls.model_info["repo_status"]),
                model_status=ModelStatus(**cls.model_info["model_status"])
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
        search_word,
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
            search_word (str): The model selection criteria.
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
      
        if search_word in CUSTOM_SEARCH_KEY:
            model_path_to_check = CUSTOM_SEARCH_KEY[search_word]
            _check_url = f"https://huggingface.co/{model_path_to_check}"
            if is_valid_url(_check_url):
                search_word = model_path_to_check
                self.model_info["model_path"] = _check_url
            else:
                logger.warning(f"The following custom search keys are ignored.`{search_word} : {CUSTOM_SEARCH_KEY[search_word]}`")

        if local_file_only:
            model_path = next(self.File_search(
                search_word=search_word,
                auto=auto
            ))
            self.model_info["model_status"]["single_file"] = True
            self.model_info["model_path"] = model_path
            self.model_info["load_type"] = "from_single_file"

        elif search_word.startswith("https://huggingface.co/"):
            if not is_valid_url(search_word):
                raise ValueError("Could not load URL")
            else:
                if download:
                    model_path = self.run_hf_download(search_word)
                else:
                    model_path = search_word

                self.model_info["model_status"]["single_file"] = True
                self.model_info["model_path"] = model_path
                repo, file_name = self.repo_name_or_path(search_word)
                if file_name:
                    self.model_info["model_status"]["filename"] = file_name
                    self.model_info["model_status"]["single_file"] = True
                    self.model_info["load_type"] = "from_single_file"
                else:
                    self.model_info["model_status"]["single_file"] = False
                    self.model_info["load_type"] = "from_pretrained"


        elif search_word.startswith("https://civitai.com/"):
            model_path = self.search_for_civitai(
                search_word=search_word,
                auto=auto,
                model_type=model_type,
                download=download,
                civitai_token=civitai_token,
                skip_error=False
            )

        elif os.path.isfile(search_word):
            model_path = search_word
            self.model_info["model_path"] = search_word
            self.model_info["model_status"]["single_file"] = True
            self.model_info["load_type"] = "from_single_file"
            self.model_info["model_status"]["local"] = True

        elif os.path.isdir(search_word):
            if os.path.exists(os.path.join(search_word, self.Config_file)):
                model_path = search_word
                self.model_info["model_path"] = search_word
                self.model_info["model_status"]["single_file"] = False
                self.model_info["load_type"] = "from_pretrained"
                self.model_info["model_status"]["local"] = True
            else:
                raise FileNotFoundError(f"model_index.json not found in {search_word}")

        elif search_word.count("/") == 1:
            creator_name, repo_name = search_word.split("/")

            if auto and self.diffusers_model_check(search_word):
                if download:
                    model_path = self.run_hf_download(search_word)
                    self.model_info["model_status"]["single_file"] = False
                else:
                    model_path = search_word
                    self.model_info["model_status"]["single_file"] = False
                self.model_info["load_type"] = "from_pretrained"

            elif auto and (not self.hf_model_check(search_word)):
                raise ValueError(f'The specified repository could not be found, please try turning off "auto" (search_word:{search_word})')
            else:
                file_path=self.file_name_set(search_word,auto,model_type)
                if file_path is None:
                    raise ValueError("Model not found")
                elif file_path == "DiffusersFormat":
                    if download:
                        model_path = self.run_hf_download(search_word)
                    else:
                        model_path = search_word

                    self.model_info["model_status"]["single_file"] = False
                    self.model_info["load_type"] = "from_pretrained"
                    
                else:
                    hf_model_path = f"https://huggingface.co/{search_word}/blob/{branch}/{file_path}"
                    
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
                model_path = self.search_for_hf(
                    search_word=search_word,
                    auto=auto,
                    model_format=model_format,
                    model_type=model_type,
                    download=download,
                    include_civitai=True
                    )
                if model_path is None:
                    model_path = self.search_for_civitai(
                        search_word=search_word,
                        auto=auto,
                        model_type=model_type,
                        download=download,
                        civitai_token=civitai_token,
                        include_hugface=False
                    )
                    if not model_path:
                        raise ValueError("No models matching the criteria were found.")
                
            else:
                model_path = self.search_for_civitai(
                    search_word=search_word,
                    auto=auto,
                    model_type=model_type,
                    download=download,
                    civitai_token=civitai_token,
                    include_hugface=True
                )
                if not model_path:
                    model_path = self.search_for_hf(
                        search_word=search_word,
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


