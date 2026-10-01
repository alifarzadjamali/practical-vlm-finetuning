import platform
from importlib.metadata import PackageNotFoundError, version

import torch


def main() -> None:
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            print(f"{package}:", version(package))
        except PackageNotFoundError:
            print(f"{package}: not installed")

    print("CUDA ready:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No GPU today. You can still prepare and check your data.")
        return

    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / 1024**3
    print("GPU:", properties.name)
    print(f"GPU memory: {memory_gb:.1f} GB")
    print("bfloat16 ready:", torch.cuda.is_bf16_supported())


if __name__ == "__main__":
    main()
