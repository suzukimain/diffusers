from dataclasses import dataclass, field

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