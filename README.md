# FLUX.2
by Black Forest Labs: https://bfl.ai.

Documentation for our API can be found here: [docs.bfl.ai](https://docs.bfl.ai/).

This repo contains minimal inference code to run image generation & editing with our FLUX.2 open-weight models.

## `FLUX.2 [dev]`

`FLUX.2 [dev]` is a 32B parameter flow matching transformer model capable of generating and editing (multiple) images. The model is released under the [FLUX.2-dev Non-Commercial License](model_licenses/LICENSE-FLUX-DEV) and can be found [here](https://huggingface.co/black-forest-labs/FLUX.2-dev).

Note that the below script for `FLUX.2 [dev]` needs considerable amount of VRAM (H100-equivalent GPU). We partnered with Hugging Face to make quantized versions that run on consumer hardware; below you can find instructions on how to run it on a RTX 4090 with a remote text encoder, for other quantization sizes and combinations, check the [diffusers quantization guide here](docs/flux2_dev_hf.md).

### Text-to-image examples

![t2i-grid](assets/teaser_generation.png)

### Editing examples

![edit-grid](assets/teaser_editing.png)

### Prompt upsampling

`FLUX.2 [dev]` benefits significantly from prompt upsampling. The inference script below offers the option to use both local prompt upsampling with the same model we use for text encoding ([`Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)), or alternatively, use any model on [OpenRouter](https://openrouter.ai/) via an API call.

See the [upsampling guide](docs/flux2_with_prompt_upsampling.md) for additional details and guidance on when to use upsampling.

## `FLUX.2` autoencoder

The FLUX.2 autoencoder has considerably improved over the [FLUX.1 autoencoder](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors). The autoencoder is released under [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found [here](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors). For more information, see our [technical blogpost](https://bfl.ai/research/representation-comparison).

## Local installation

The inference code was tested on GB200 and H100 (with CPU offloading).

### GB200

On GB200, we tested `FLUX.2 [dev]` using CUDA 12.9 and Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

### H100

On H100, we tested `FLUX.2 [dev]` using CUDA 12.6 and Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
```

## Run the CLI

Before running the CLI, you may download the weights from [here](https://huggingface.co/black-forest-labs/FLUX.2-dev) and set the following environment variables.

```bash
export FLUX2_MODEL_PATH="<flux2_path>"
export AE_MODEL_PATH="<ae_path>"
```

If you don't set the environment variables, the weights will be downloaded
automatically.

You can start an interactive session with loaded weights by running the
following command. That will allow you to do both text to image generation as
well as editing one or multiple images.
```bash
export PYTHONPATH=src
python scripts/cli.py
```

On H100, we additionally set the flag `--cpu_offloading True`.

## Watermarking

We've added an option to embed invisible watermarks directly into the generated images
via the [invisible watermark library](https://github.com/ShieldMnt/invisible-watermark).

Additionally, we are recommending implementing a solution to mark the metadata of your outputs, such as [C2PA](https://c2pa.org/)

## 🧨 Lower VRAM diffusers example

The below example should run on a RTX 4090. For more examples check the [diffusers quantization guide here](docs/flux2_dev_hf.md)

```python
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png") #optional image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@misc{flux-2-2025,
    author={Black Forest Labs},
    title={{FLUX.2: Frontier Visual Intelligence}},
    year={2025},
    howpublished={\url{https://bfl.ai/blog/flux-2}},
}
```
