import re

import torch

# Get a list of all available PyTorch operations
torch_operations = set(dir(torch))


# Function to modify the input string
def modify_operations(input_string):
    # Regular expression to find function names and calls
    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("

    # Function to replace matches with 'torch.' prefix if they are PyTorch operations
    def replacer(match):
        func_name = match.group(1)
        if func_name in torch_operations:
            return f"torch.{func_name}("
        return match.group(0)

    # Apply the replacement
    modified_string = re.sub(pattern, replacer, input_string)
    return modified_string
