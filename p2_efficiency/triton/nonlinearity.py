import torch
import triton
import triton.language as tl

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    num_elements = x.numel()
    block_size = 1024 # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    # Kernel launch
    triton_gelu_kernel[(num_blocks, )](x, y, num_elements, BLOCK_SIZE = block_size)

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at 'x_ptr' and output is at 'y_ptr'
    #    |     Block 0     |     Block 1     |       ...       |
    #                  BLOCK_SIZE                       num_elements

    pid = tl.program_id(axis = 0) # id of the block
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    offsets = block_start + tk.arange(0, BLOCK_SIZE)

    # Handle boundaries
    mask = offsets < num_elements

    # 1. Read input
    x = tl.load(x_ptr + offsets, mask = mask)

    # 2. Calculations are here
    # Approx gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh) # temporary value

    # 3. Store outputs
    tl.store(y_ptr + offsets, y, mask = mask)

