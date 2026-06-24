import os

import torch


def print_cuda_diagnostics():
    print("Python-visible CUDA diagnostics")
    print("=" * 80)

    print(f"torch.__version__        : {torch.__version__}")
    print(f"torch.version.cuda       : {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(
        f"CUDA_VISIBLE_DEVICES     : {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )

    if not torch.cuda.is_available():
        print("\nNo CUDA device is visible to PyTorch.")
        print("If you have a GPU, make sure PyTorch is installed with CUDA support.")
        print(
            "On a cluster, make sure you are inside a GPU allocation, not just on a login node."
        )
        return

    current = torch.cuda.current_device()
    print(f"torch.cuda.current_device(): {current}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        print("\n" + "-" * 80)
        print(f"Logical device index : cuda:{i}")
        print(f"Name                 : {props.name}")
        print(f"Compute capability   : {props.major}.{props.minor}")
        print(f"Total memory         : {props.total_memory / 1024**3:.2f} GiB")
        print(f"Multiprocessors      : {props.multi_processor_count}")

        # These require CUDA runtime access to the device.
        free_b, total_b = torch.cuda.mem_get_info(i)
        print(f"Runtime free memory  : {free_b / 1024**3:.2f} GiB")
        print(f"Runtime total memory : {total_b / 1024**3:.2f} GiB")

        print(
            f"Allocated by PyTorch : {torch.cuda.memory_allocated(i) / 1024**2:.2f} MiB"
        )
        print(
            f"Reserved by PyTorch  : {torch.cuda.memory_reserved(i) / 1024**2:.2f} MiB"
        )
