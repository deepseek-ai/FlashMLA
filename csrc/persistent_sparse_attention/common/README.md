# Native common utilities

Only implementation-neutral CUDA/C++ helpers shared by multiple operations
belong here. Kernel implementations and stage-specific constants belong in
their canonical stage directory.

Common headers must not:

* import files from sibling repositories;
* encode historical revision names;
* own global CUDA state;
* assume a non-current PyTorch CUDA stream; or
* allocate memory during CUDA Graph replay.
