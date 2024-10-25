from dataclasses import dataclass


class DataConfig:
    Config_file: str = "model_index.json"
    exts: list =  [".safetensors", ".ckpt",".bin"]

    model_dict: dict = {
        "StableDiffusion" : "stabilityai/stable-diffusion-2-1",
        "waifu diffusion": "hakurei/waifu-diffusion",
        "Anything-v3.0": "Linaqruf/anything-v3.0",
        "anything-midjourney-v-4-1": "Joeythemonster/anything-midjourney-v-4-1",
        "Anything-v4.5": "shibal1/anything-v4.5-clone",
        "AB4.5_AC0.2": "aioe/AB4.5_AC0.2",
        "basil_mix": "nuigurumi/basil_mix",
        "Waifu-Diffusers": "Nilaier/Waifu-Diffusers",
        "Double-Exposure-Diffusion": "joachimsallstrom/Double-Exposure-Diffusion",
        "openjourney-v4": "prompthero/openjourney-v4",
        "ACertainThing": "JosephusCheung/ACertainThing",
        "Counterfeit-V2.0": "gsdf/Counterfeit-V2.0",
        "Counterfeit-V2.5": "gsdf/Counterfeit-V2.5",
        "chilled_remix":"sazyou-roukaku/chilled_remix",
        "chilled_reversemix":"sazyou-roukaku/chilled_reversemix",
        "7th_Layer": "syaimu/7th_test",
        "loli": "JosefJilek/loliDiffusion",
        "EimisAnimeDiffusion_1.0v": "eimiss/EimisAnimeDiffusion_1.0v",
        "JWST-Deep-Space-diffusion" : "dallinmackay/JWST-Deep-Space-diffusion",
        "Riga_Collection": "natsusakiyomi/Riga_Collection",
        "sd-db-epic-space-machine" : "rabidgremlin/sd-db-epic-space-machine",
        "spacemidj" : "Falah/spacemidj",
        "anime-kawai-diffusion": "Ojimi/anime-kawai-diffusion",
        "Realistic_Vision_V2.0": "SG161222/Realistic_Vision_V2.0",
        "nasa-space-v2" : "sd-dreambooth-library/nasa-space-v2-768",
        "meinamix_meinaV10": "namvuong96/civit_meinamix_meinaV10",
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
        

@dataclass
class RepoStatus:
    repo_name: str = ""
    repo_id: str = ""
    version_id: str = ""


@dataclass
class ModelStatus:
    search_word: str = ""
    download_url: str = ""
    filename: str = ""
    file_id: str = ""
    fp: str = ""
    local: bool = False
    single_file: bool = False


@dataclass
class ModelData:
    """
    MAP:
       {"model_path": "",
        "load_type": "",
        "repo_status": {
            "repo_name": "",
            "repo_id": "",
            "version_id": ""
            },
        "model_status": {
            "search_word": "",
            "download_url": "",
            "filename": "",
            "file_id": "",
            "fp": "",
            "local": False,
            "single_file": False
            }
       }
    """
    model_path: str = ""
    load_type: str = ""  # "" or "from_single_file" or "from_pretrained"
    repo_status: RepoStatus = RepoStatus()
    model_status: ModelStatus = ModelStatus()
