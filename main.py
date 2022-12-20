import numpy as np
import cupy as cp

memory = cp.get_default_memory_pool()
pinned_memory = cp.get_default_pinned_memory_pool()
print(cp.get_default_memory_pool().get_limit())
print(pinned_memory.n_free_blocks())
a = cp.zeros((100,100))
print(a.nbytes)
print(memory.total_bytes())

if __name__ == "__main__":
    pass