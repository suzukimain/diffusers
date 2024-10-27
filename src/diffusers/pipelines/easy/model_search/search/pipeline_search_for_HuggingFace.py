import re
import requests
from requests import HTTPError

from dataclasses import asdict
from huggingface_hub import (
    hf_hub_download, 
    HfApi,
    login
    )

from ....pipeline_utils import DiffusionPipeline
from .....loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    is_valid_url
    )

from ..search_utils.search_pipeline_utils import SearchPipelineConfig
from ..search_utils.model_search_data_classes import ModelData


class HFSearchPipeline(SearchPipelineConfig):
    """
    Huggingface class is used to search and download models from Huggingface.

    Attributes:
        num_prints (int): Number of prints.
        model_id (str): Model ID.
        model_name (str): Model name.
        vae_name (str): VAE name.
        model_file (str): Model file.
        diffuser_model (bool): Diffuser model flag.
        check_choice_key (str): Check choice key.
        choice_number (int): Choice number.
        special_file (str): Special file.
        hf_repo_id (str): Huggingface repository ID.
        force_download (bool): Force download flag.
        hf_api (HfApi): Huggingface API instance.

    Methods:
        __call__(*args, **kwds): Returns model data.
        repo_name_or_path(model_name_or_path): Returns repository name or path.
        hf_login(token): Logs in to Huggingface.
        _hf_repo_download(path, branch): Downloads repository.
        run_hf_download(url_or_path, branch): Downloads model from Huggingface.
        model_safe_check(model_list): Checks if model is safe.
        list_safe_sort(model_list): Sorts model list by safety.
        diffusers_model_check(checked_model, branch): Checks if model is a diffuser model.
        hf_model_check(path): Checks if model exists on Huggingface.
        model_data_get(path, model_info): Retrieves model data.
        hf_model_search(model_path, limit_num): Searches for models on Huggingface.
        old_hf_model_search(model_path, limit_num): Old method for searching models on Huggingface.
        hf_model_info(model_name): Retrieves model information from Huggingface.
        old_hf_model_info(model_select): Old method for retrieving model information from Huggingface.
        hf_security_check(check_dict): Checks model security.
        check_if_file_exists(hf_repo_info): Checks if file exists in Huggingface repository.
        hf_models(model_name, limit): Retrieves models from Huggingface.
        find_max_like(model_dict_list): Finds the model with the most likes.
        sort_by_likes(model_dict_list): Sorts models by likes.
        get_hf_files(check_data): Retrieves files from Huggingface.
        model_name_search(model_name, auto_set, model_format, Recursive_execution, extra_limit, include_civitai):
            Searches for model name on Huggingface.
        file_name_set_sub(model_select, file_value): Sets file name.
        file_name_set(model_select, auto, model_format, model_type): Sets file name.
    """

    def __init__(self):
        super().__init__()
        self.num_prints = 20
        self.model_id = ""
        self.model_name = ""
        self.vae_name = ""
        self.model_file = ""
        self.diffuser_model = False
        self.check_choice_key = ""
        self.choice_number = -1
        self.special_file = ""
        self.hf_repo_id = ""
        self.force_download = False
        self.hf_api = HfApi()


    def __call__(self, **keywords):
        return self.hf_model_set(**keywords)
    

    def hf_model_set(
            self,
            model_select,
            auto,
            model_format,
            model_type,
            download,
            include_civitai=True
            ):
        
        model_path = ""
        model_name = self.model_name_search(
            model_name=model_select,
            auto_set=auto,
            model_format=model_format,
            include_civitai=include_civitai
            )
        if not model_name == "_hf_no_model":
            file_path = self.file_name_set(
                model_select=model_name,
                auto=auto,
                model_format=model_format,
                model_type=model_type
                )
            if file_path == "_DFmodel":
                if download:
                    model_path = self.run_hf_download(
                        model_name,
                        branch=self.branch
                        )
                else:
                    model_path = model_name
                self.model_data["model_path"] = model_path
                self.model_data["model_status"]["single_file"] = False
                self.model_data["load_type"] = "from_pretrained"

            else:
                hf_model_path = f"https://huggingface.co/{model_name}/blob/{self.branch}/{file_path}"
                if download:
                    model_path = self.run_hf_download(hf_model_path)
                else:
                    model_path = hf_model_path
                self.model_data["model_status"]["single_file"] = True
                self.model_data["load_type"] = "from_single_file"
                self.model_data["model_status"]["filename"] = file_path

            return model_path
        else:
            return "_hf_no_model"


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
                raise HTTPError(f"Invalid URL: {url_or_path}")
            hf_path, file_name = self.repo_name_or_path(url_or_path)
            self.logger.debug(f"url_or_path:{url_or_path}")
            self.logger.debug(f"hf_path: {hf_path} \nfile_name: {file_name}")
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
                    raise HTTPError("Invalid hf_path")
            else:
                raise TypeError("Invalid path_or_url")
        elif self.diffusers_model_check(url_or_path):
            self.logger.debug(f"url_or_path: {url_or_path}")
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
                if bool(re.search(r"(?i)[-ー_＿](sfw|safe)", check_model)):
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
            if bool(re.search(r"(?i)[-ー_＿](sfw|safe)", check_model)):
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
        return is_valid_url(f"https://huggingface.co/{path}")


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
                any(file_path.endswith(ext) for ext in self.exts)
                and not any(file_path.endswith(ex) for ex in self.exclude)
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
        return [asdict(value) for value in list(self.hf_api.list_models(**params))]


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
        hf_info = self.hf_api.model_info(
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
            raise HTTPError("A hugface login or token is required")
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
                for ext in self.exts
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
                    any(fi_path.endswith(ext) for ext in self.exts)
                    and (not any(fi_path.endswith(ex) for ex in self.exclude))
                    and (not any(fi_path.endswith(st) for st in self.config_file_list))
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

        previous_model_selection = self.check_func_hist(
            key="hf_model_name", return_value=True
        )
        models_to_exclude: list = self.check_func_hist(
            key="dangerous_model", return_value=True, missing_value=[]
        )

        repo_model_list = [
            model
            for model in original_repo_model_list
            if model["model_id"] not in models_to_exclude
        ]

        if not auto_set:
            print("\033[34mThe following model paths were found\033[0m")
            if previous_model_selection is not None:
                print(f"\033[34mPrevious Choice: {previous_model_selection}\033[0m")
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
                    return "_hf_no_model"
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

                    security_risk = self.model_data_get(path=choice_path)[
                        "security_risk"
                    ]
                    if security_risk == 2:
                        print("\033[31mThis model has a security problem。\033[0m")
                        if choice_path not in models_to_exclude:
                            models_to_exclude.append(choice_path)
                        self.update_json_dict(
                            key="dangerous_model", value=models_to_exclude
                        )
                        continue
                    else:
                        if security_risk == 1:
                            self.logger.warning(
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
                        self.logger.warning(
                            "No models in diffusers format were found."
                        )
                        choice_path = "_hf_no_model"
            else:
                choice_path = "_hf_no_model"

        return choice_path


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
                    return "_hf_no_model"
                else:
                    raise ValueError(
                        "No available files were found in the specified repository"
                    )
            else:
                print("\033[34mOnly models in Diffusers format found\033[0m")
                while True:
                    result = input("Do you want to use it?[y/n]: ")
                    if result.lower() in ["y", "yes"]:
                        return "_DFmodel"
                    elif result.lower() in ["n", "no"]:
                        sec_result = input("Searching for civitai?[y/n]: ")
                        if sec_result.lower() in ["y", "yes"]:
                            return "_hf_no_model"
                        elif sec_result.lower() in ["n", "no"]:
                            raise ValueError(
                                "Processing was stopped because no corresponding model was found."
                            )
                    else:
                        print("\033[34mPlease enter only [y,n]\033[0m")

        file_value = self.list_safe_sort(file_value)
        if len(file_value) >= self.num_prints:
            start_number = "1"
            choice_history = self.check_func_hist(key=check_key, return_value=True)
            if choice_history:
                if choice_history > self.num_prints + 1:
                    choice_history = self.num_prints + 1
                print(f"\033[33m＊Previous number: {choice_history}\033[0m")

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
                    self.choice_number = -1
                    self.update_json_dict(key=check_key, value=choice)
                    return "_DFmodel"

                elif choice == (self.num_prints + 1):
                    break
                elif 1 <= choice <= self.num_prints:
                    choice_path = file_value[choice - 1]
                    self.choice_number = choice
                    self.update_json_dict(key=check_key, value=choice)
                    return choice_path
                else:
                    print(
                        f"\033[33mPlease enter numbers from 1~{self.num_prints}\033[0m"
                    )
            print("\n\n")

        choice_history = self.check_func_hist(key=check_key, return_value=True)
        if choice_history:
            print(f"\033[33m＊Previous number: {choice_history}\033[0m")

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
                    self.choice_number = -1
                    self.update_json_dict(key=check_key, value=choice)
                    return "_DFmodel"

                if 1 <= choice <= len(file_value):
                    choice_path = file_value[choice - 1]
                    self.choice_number = choice
                    self.update_json_dict(key=check_key, value=choice)
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
            raise ValueError("No available file was found.\nPlease check the name.")
        if file_value:
            file_value = self.sort_by_version(file_value)
            if not auto:
                print("\033[34mThe following model files were found\033[0m")
                choice_path = self.file_name_set_sub(model_select, file_value)
            else:
                if self.diffuser_model and (not skip_difusers):
                    choice_path = "_DFmodel"
                else:
                    choice_path = self.model_safe_check(file_value)

        elif self.diffuser_model:
            choice_path = "_DFmodel"
        else:
            raise FileNotFoundError(
                "No available files found in the specified repository"
            )
        return choice_path