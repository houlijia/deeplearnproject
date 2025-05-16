import torch

# Parallel execution
def print_torch_threads():
    count = torch.get_num_threads()
    print("Number of threads: ", count)

def set_torch_threads(count):
    torch.set_num_threads(count)
    print("Number of threads set to: ", count)


print_torch_threads()
set_torch_threads(2)
print_torch_threads()


