from dataclasses import dataclass


class DataConfig:
    """
    Configuration class for data handling in the model search pipeline.

    Attributes:
        Config_file (str): The name of the configuration file.
        exts (list): List of supported file extensions.
        model_dict (dict): Dictionary mapping model names to their corresponding paths.
        exclude (list): List of files to exclude from the search.
        Auto_pipe_class (list): List of auto pipeline classes.
        Error_M1 (str): Error message for invalid URL format.
        config_file_list (list): List of configuration files.
    """
    Config_file: str = "model_index.json"
    exts: list =  [".safetensors", ".ckpt",".bin"]
    model_data: dict = {}

    model_info: dict = {}

    model_dict: dict = {
        "sd" : "stabilityai/stable-diffusion-2-1",
        }
    
    exclude: list =  [
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

    Auto_pipe_class: list = [
        "AutoPipelineForText2Image",
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        ]

    Error_M1: str = (
        '''
        Could not load URL.
        Format:"https://huggingface.co/<creator_name>/<repo_name>/blob/main/<path_to_file>"
        EX1: "https://huggingface.co/gsdf/Counterfeit-V3.0/blob/main/Counterfeit-V3.0.safetensors"
        EX2: "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
        '''
        )

    config_file_list: list = [
        "preprocessor_config.json",
        "config.json",
        "model.fp16.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.fp16.bin",
        "scheduler_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.fp16.bin",
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.non_ema.bin",
        "diffusion_pytorch_model.non_ema.safetensors",
        "diffusion_pytorch_model.safetensors",
        ]
    
    def __init__(self):
        self.single_file_only: bool  = False
        self.hf_token: str = None
        self.force_download: bool = False
        self.model_info = {
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