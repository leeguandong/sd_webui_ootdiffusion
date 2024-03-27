"""Install requirements for WD14-tagger."""
import os
import sys

from launch import run  # pylint: disable=import-error

NAME = "OOTDiffusion"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")

# 权重下载
humanparsing = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints/humanparsing")
ootd = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints/ootd")

run(f"mkdir {humanparsing}")
run(f"cd {humanparsing}")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx")

run(f"mkdir {ootd}")
run(f"cd {ootd}")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/model_index.json")

# 自己下载一下权重
# run("git clone https://huggingface.co/levihsu/OOTDiffusion")
# %cd /content
# !git clone -b dev https://github.com/camenduru/OOTDiffusion
# %cd /content/OOTDiffusion
#
# !apt -y install -qq aria2
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/images/demo.png -d /content/OOTDiffusion/images -o demo.png
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth -d /content/OOTDiffusion/checkpoints/humanparsing -o exp-schp-201908261155-lip.pth
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908301523-atr.pth -d /content/OOTDiffusion/checkpoints/humanparsing -o exp-schp-201908301523-atr.pth
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/openpose/ckpts/body_pose_model.pth -d /content/OOTDiffusion/checkpoints/openpose/ckpts -o body_pose_model.pth
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/feature_extractor/preprocessor_config.json -d /content/OOTDiffusion/checkpoints/ootd/feature_extractor -o preprocessor_config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/config.json -d /content/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm -o config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors -d /content/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm -o diffusion_pytorch_model.safetensors
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/config.json -d /content/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton -o config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors -d /content/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton -o diffusion_pytorch_model.safetensors
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/scheduler/scheduler_config.json -d /content/OOTDiffusion/checkpoints/ootd/scheduler -o scheduler_config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/text_encoder/config.json -d /content/OOTDiffusion/checkpoints/ootd/text_encoder -o config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/text_encoder/pytorch_model.bin -d /content/OOTDiffusion/checkpoints/ootd/text_encoder -o pytorch_model.bin
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/merges.txt -d /content/OOTDiffusion/checkpoints/ootd/tokenizer -o merges.txt
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/special_tokens_map.json -d /content/OOTDiffusion/checkpoints/ootd/tokenizer -o special_tokens_map.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/tokenizer_config.json -d /content/OOTDiffusion/checkpoints/ootd/tokenizer -o tokenizer_config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/vocab.json -d /content/OOTDiffusion/checkpoints/ootd/tokenizer -o vocab.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/vae/config.json -d /content/OOTDiffusion/checkpoints/ootd/vae -o config.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/vae/diffusion_pytorch_model.bin -d /content/OOTDiffusion/checkpoints/ootd/vae -o diffusion_pytorch_model.bin
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/model_index.json -d /content/OOTDiffusion/checkpoints/ootd -o model_index.json
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SUPIR/resolve/main/clip-vit-large-patch14.tar -d /content/OOTDiffusion/checkpoints -o clip-vit-large-patch14.tar
#
# !pip install -q gradio config einops ninja diffusers==0.24.0 accelerate==0.26.1
#
# %cd /content/OOTDiffusion/checkpoints
# !mkdir clip-vit-large-patch14
# %cd /content/OOTDiffusion/checkpoints/clip-vit-large-patch14
# !tar -xvf ../clip-vit-large-patch14.tar
#
# %cd /content/OOTDiffusion/run
# !python gradio_ootd.py