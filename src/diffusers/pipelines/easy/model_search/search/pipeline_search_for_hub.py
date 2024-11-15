import os
from typing import Union

from .....utils import logging
from .....loaders.single_file_utils import is_valid_url

from huggingface_hub import login
from .pipeline_search_for_HuggingFace import HFSearchPipeline
from .pipeline_search_for_Civitai import CivitaiSearchPipeline
from ..search_utils import (
    RepoStatus,
    ModelStatus,
    SearchPipelineOutput,
    CONFIG_FILE_LIST,
    CUSTOM_SEARCH_KEY,
    EXTENSION
    )


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        self.update_json_dict(
            key="model_path",
            value=model_path
        )       
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
        return cls().search_for_hubs(cls, search_word=search_word, **kwargs)
    
    @classmethod
    def for_HF(
        cls,
        search_word,
        **kwargs
    ):
        return cls().search_for_hf(cls, search_word=search_word, **kwargs)
    
    @classmethod
    def for_civitai(
        cls,
        search_word,
        **kwargs
    ):
        return cls().search_for_civitai(cls, search_word=search_word, **kwargs)
