import os
import requests
from requests import HTTPError
from tqdm.auto import tqdm

from .....utils import logging
from .....loaders.single_file_utils import is_valid_url

from ..search_utils import (
    SearchPipelineConfig,
    SearchPipelineOutput,
    RepoStatus,
    ModelStatus
    )


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CivitaiSearchPipeline(SearchPipelineConfig):
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
            raise HTTPError(f"Could not get elements from the URL: {err}")
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
