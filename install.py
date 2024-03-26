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
humanparsing = os.path.join(os.path.dirname(os.path.realpath(__file__)), "humanparsing")
ootd = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ootd")

run(f"mkdir {humanparsing}")
run(f"cd {humanparsing}")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx")

run(f"mkdir {ootd}")
run(f"cd {ootd}")
run("wget https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/model_index.json")
