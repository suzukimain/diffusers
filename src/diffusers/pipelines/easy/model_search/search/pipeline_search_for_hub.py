import os

from .....utils import logging
from .....loaders.single_file_utils import is_valid_url

from .mix_class import Config_Mix
from ..search_utils import (
    ModelData,
    RepoStatus,
    ModelStatus
    )


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ModelSearchPipeline(Config_Mix):
    def __init__(self):
        super().__init__()
    

    def __call__(
            self,
            seach_word,
            auto=True,
            download=False,
            model_type="Checkpoint",
            model_format = "single_file",
            branch = "main",
            priority_hub = "hugface",
            local_file_only = False,
            hf_token = None,
            civitai_token = None,
            include_params = False,
            ):

        self.single_file_only = True if "single_file" == model_format else False

        self.model_info["model_status"]["search_word"] = seach_word
        self.model_info["model_status"]["local"] = True if download or local_file_only else False

        self.hf_login(hf_token)

        result = self.model_set(
            model_select = seach_word,
            auto = auto,
            download = download,
            model_format = model_format,
            model_type = model_type,
            branch = branch,
            priority_hub = priority_hub,
            local_file_only = local_file_only,
            civitai_token = civitai_token,
            include_params = include_params
        )
        if include_params:
            return self.SearchPipelineOutput(self.model_info)
        else:
            return result        
        

    def File_search(
            self,
            search_word,
            auto = True
            ):
        """
        only single file
        """
        closest_match = None
        closest_distance = float('inf')
        for root, dirs, files in os.walk("/"):
            for file in files:
                if any(file.endswith(ext) for ext in self.exts):
                    path = os.path.join(root, file)
                    if path not in self.exclude:
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
    

    def user_select_file(self, search_word):
        """
        Allow user to select a file from the search results.
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
        """
        return sum(1 for a, b in zip(search_word, file_name) if a != b)  


    def model_set(
            self,
            model_select,
            auto = True,
            download = False,
            model_format = "single_file",
            model_type = "Checkpoint",
            branch = "main",
            priority_hub = "hugface",
            local_file_only = False,
            civitai_token = None,
            include_params = False
            ):
        """
        parameter:
        model_format:
            one of the following: "all","diffusers","single_file"
        return:
        if path_only is false
        [model_path:str, {base_model_path: str,single_file: bool}]
        """

        if not model_type  in ["Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses"]:
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

        if local_file_only:
            model_path = next(self.File_search(
                search_word = model_select,
                auto = auto
                ))
            self.model_info["model_status"]["single_file"] = True
            self.model_info["model_path"] = model_path
            self.model_info["load_type"] = "from_single_file"

        elif model_select.startswith("https://huggingface.co/"):
            if not is_valid_url(model_select):
                raise ValueError(self.Error_M1)
            else:
                if download:
                    model_path = self.run_hf_download(model_select)
                else:
                    model_path = model_select

                self.model_info["model_status"]["single_file"] = True
                self.model_info["model_path"] = model_path
                repo,file_name = self.repo_name_or_path(model_select)
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
            if os.path.exists(os.path.join(model_select,self.Config_file)):
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
                if file_path == "_hf_no_model":
                    raise ValueError("Model not found")
                elif file_path == "_DFmodel":
                    if download:
                        model_path= self.run_hf_download(model_select)
                    else:
                        model_path = model_select

                    self.model_info["model_status"]["single_file"] = False
                    self.model_info["load_type"] = "from_pretrained"
                    
                else:
                    hf_model_path=f"https://huggingface.co/{model_select}/blob/{branch}/{file_path}"
                    
                    if download:
                        model_path = self.run_hf_download(hf_model_path)
                    else:
                        model_path = hf_model_path
                    
                    self.model_info["model_status"]["filename"] = file_path
                    self.model_info["model_status"]["single_file"] = True
                    self.model_info["load_type"] = "from_single_file"
                
            self.model_info["repo_status"]["repo_name"] = repo_name
                
        else:
            if priority_hub == "hugface":
                model_path = self.hf_model_set(
                    model_select=model_select,
                    auto=auto,
                    model_format=model_format,
                    model_type=model_type,
                    download=download,
                    include_civitai=True
                    )
                if model_path == "_hf_no_model":
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
                        model_select = model_select,
                        auto = auto,
                        model_format=model_format,
                        model_type=model_type,
                        download=download,
                        include_civitai=False
                        )
                    if model_path == "_hf_no_model":
                        raise ValueError("No models matching the criteria were found.")
                
        self.model_info["model_path"] = model_path
        self.update_json_dict(
            key = "model_path",
            value = model_path
            )       
        if include_params:
            yield self.model_info
        else:
            yield model_path

    @classmethod
    def from_hub(cls, model_id, **kwargs):
        """
        Load a pipeline from the Hugging Face Hub.

        Args:
            model_id (str): The model ID on the Hugging Face Hub.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelSearchPipeline: The loaded pipeline.
        """
        instance = cls()
        instance.model_data["model_status"]["search_word"] = model_id
        instance.model_data["model_status"]["local"] = False
        instance.model_data["model_path"] = f"https://huggingface.co/{model_id}"
        return instance

    @classmethod
    def from_hf(cls, model_id, **kwargs):
        """
        Load a pipeline from Hugging Face.

        Args:
            model_id (str): The model ID on Hugging Face.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelSearchPipeline: The loaded pipeline.
        """
        pass

    @classmethod
    def from_civitai(cls, model_id, **kwargs):
        """
        Load a pipeline from Civitai.

        Args:
            model_id (str): The model ID on Civitai.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelSearchPipeline: The loaded pipeline.
        """
        pass