"""Run in the environment you trained in to pin exact versions:  python make_requirements.py"""
import importlib.metadata as md
pkgs = ["torch", "torchvision", "segmentation-models-pytorch", "albumentations", "opencv-python-headless",
        "pycocotools", "numpy", "pandas", "kagglehub", "tqdm", "matplotlib"]
out = []
for p in pkgs:
    try: out.append(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError: print("missing:", p)
open("requirements.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
