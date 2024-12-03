# Search models on Civitai and Hugging Face

The [auto_diffusers](https://github.com/suzukimain/auto_diffusers) library provides additional functionalities to Diffusers such as searching for models on Civitai and the Hugging Face Hub.
Please refer to the original library [here](https://pypi.org/project/auto-diffusers/)

## Installation

Before running the scripts, make sure to install the library's training dependencies:

> [!IMPORTANT]
> To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the installation up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment.

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
##  Search for models
```bash
!wget https://raw.githubusercontent.com/suzukimain/diffusers/refs/heads/ModelSearch/examples/model_search/search_for_civitai_and_HF.py
```

### Search for Civitai
```python
from pipeline_easy import (
    EasyPipelineForText2Image,
    EasyPipelineForImage2Image,
    EasyPipelineForInpainting,
)

# Text-to-Image
pipeline = EasyPipelineForText2Image.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")


# Image-to-Image
pipeline = EasyPipelineForImage2Image.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")


# Inpainting
pipeline = EasyPipelineForInpainting.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")
```

### Search for Hugging Face
```python
from pipeline_easy import (
    EasyPipelineForText2Image,
    EasyPipelineForImage2Image,
    EasyPipelineForInpainting,
)

# Text-to-Image
pipeline = EasyPipelineForText2Image.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")


# Image-to-Image
pipeline = EasyPipelineForImage2Image.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")


# Inpainting
pipeline = EasyPipelineForInpainting.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")
```

### Arguments of `EasyPipeline.from_civitai`

| Name            | Type   | Default       | Description                                                                         |
|:---------------:|:------:|:-------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string | ー            | The search query string. Can be a keyword, Civitai URL, local directory or file path. |
| model_type      | string | `Checkpoint`  | The type of model to search for. [Details](#model_type)                             |
| base_model      | string | None          | Trained model tag (example:  `SD 1.5`, `SD 3.5`, `SDXL 1.0`)                        |
| force_download  | bool   | False         | Whether to force the download if the model already exists.                          |
| torch_dtype     | string, torch.dtype | None   | Override the default `torch.dtype` and load the model with another dtype.           |
| cache_dir       | string, Path | None         | Path to the folder where cached files are stored.                                   |
| resume          | bool   | False         | Whether to resume an incomplete download.                                           |
| token           | string | None          | API token for Civitai authentication.                                               |
| skip_error      | bool   | False         | Whether to skip errors and return None.                                             |






### Arguments of `EasyPipeline.from_huggingface`

| Name                  | Type                            | Default        | Input Available   | Description                                                                                                          |
|:---------------------:|:------------------------------:|:--------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------:|
| pretrained_model_or_path | str or os.PathLike            | ー             | ー                | Keywords to search models                                                                                            |
| checkpoint_format     | string                          | "single_file"  | `single_file`,<br>`diffusers`,<br>`all` | The format of the model checkpoint.                                                             |
| pipeline_tag          | string                          | None           | ー                 | Tag to filter models by pipeline.                                                                                    |
| torch_dtype           | str or torch.dtype              | None           | ー                 | Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the dtype is automatically derived from the model's weights. |
| force_download        | bool                            | False          | ー                 | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | str, os.PathLike         | None           | ー                 | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.   |
| token                 | str or bool                     | None           | ー                 | The token to use as HTTP bearer authorization for remote files.                                                      |



  
> Arguments of `HFSearchPipeline.for_HF`

| Name             | Type    | Default       | Description                                                   |
|:----------------:|:-------:|:-------------:|:-------------------------------------------------------------:|
| search_word      | string  | ー            | The search query string.                                      |
| revision         | string  | None          | The specific version of the model to download.                |
| checkpoint_format| string  | "single_file" | The format of the model checkpoint.                           |
| download         | bool    | False         | Whether to download the model.                                |
| force_download   | bool    | False         | Whether to force the download if the model already exists.    |
| include_params   | bool    | False         | Whether to include parameters in the returned data.           |
| pipeline_tag     | string  | None          | Tag to filter models by pipeline.                             |
| hf_token         | string  | None          | API token for Hugging Face authentication.                    |
| skip_error       | bool    | False         | Whether to skip errors and return None.                       |



### CivitaiSearchPipeline.for_civitai parameters
| Name             | Type    | Default       | Description                                                   |
|:----------------:|:-------:|:-------------:|:-------------------------------------------------------------:|
| search_word      | string  | ー            | The search query string. Can be a keyword, Hugging Face or Civitai URL, local directory or file path, or a Hugging Face path (`<creator>/<repo>`).                                      |
| model_type       | string  | "Checkpoint"  | The type of model to search for.                              |
| base_model       | string  | None          | The base model to filter by.                                  |
| download         | bool    | False         | Whether to download the model.                                |
| force_download   | bool    | False         | Whether to force the download if the model already exists.    |
| civitai_token    | string  | None          | API token for Civitai authentication.                         |
| include_params   | bool    | False         | Whether to include parameters in the returned data.           |
| skip_error       | bool    | False         | Whether to skip errors and return None.                       |



<a id="search-word"></a>
<details open>
<summary>search_word</summary>

| Type                         | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| keyword                      | Keywords to search model<br>                                           |
| url                          | URL of either huggingface or Civitai                                   |
| Local directory or file path | Locally stored model paths                                             |
| huggingface path             | The following format: `< creator > / < repo >`                         |

</details>


<a id="model_type"></a>
<details open>
<summary>model_type</summary>

| Input Available              |
| :--------------------------: | 
| `Checkpoint`                 | 
| `TextualInversion`           |
| `Hypernetwork`               |
| `AestheticGradient`          |
| `LORA`                       |
| `Controlnet`                 |
| `Poses`                      |

</details>


<a id="checkpoint_format"></a>
<details open>
<summary>checkpoint_format</summary>

| Argument                     | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| all                          | The `multifolder diffusers format checkpoint` takes precedence.        |                                      
| single_file                  | Only `single file checkpoint` are searched.                            |
| diffusers                    | Search only for `multifolder diffusers format checkpoint`              |

</details>

</details>
