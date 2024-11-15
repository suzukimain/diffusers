from dataclasses import dataclass

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