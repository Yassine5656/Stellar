# Stellar

**Stellar** is a simplified **Large Language Model** (LLM) (*work-in-progress*), made from scratch and written in **C++20**, using OpenMP for parallelized computations.
This project focuses on deep-learning, starting with a Tensor Library with an emphasis on CPU optimization.

> ***Note:** This project is in its early stage. Only the tensor core is implemented, no neural layer, autograd or LLM components are available yet.
Besides, the project currently uses OpenMP. It may change in a future update.*

## Key features

### Tensor Core

- **Memory Efficient Management:** Contiguous Row-major data structure to maximize cache use and minimize allocation overhead.
- **Hybrid Parallelism:** Leverages **OpenMP** for accelerated computations. The Tensor Core employs ```PARALLEL_THRESHOLD``` strategy to ensure an optimal performance trade-off between threading overhead and compute speed.
- **Weight Initialization:** Includes native implementations of industry-standard initialization, including **Xavier Uniform** and **He Normal**.
- **Persistence:** Efficient local persistence designed for I/O efficiency during save and load.

## Architecture

```text
stellar/
├── include/                      # API includes (headers)
├── libstellar/                   # LLM Core implementation
│   ├── include/                  # Private headers
│   └── core/
├── src/                          # Application EntryPoint (CLI)
├── tests/                        # Unit tests
└── CMakeLists.txt
```

## Getting Started

### Prerequisites

>- ***Compiler:*** GCC 10+ or Clang 10+ (C++20 required)
>- ***Build System:*** CMake 3.20+
>- ***Libraries:*** OpenMP

----------

### Installation

```bash
# Clone the repository
git clone git@github.com:Yassine5656/Stellar.git Stellar
cd Stellar
```

### Build Instructions

```bash
# Configure the project in Release mode for maximal performance
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Build project
cmake --build build --config Release -j$(nproc)
```

### Running Application

```bash
./build/stellar
```

## Roadmap

### Short-Term

- [x] **~~Basic Tensor & Row Major Layout~~**
- [x] **~~OpenMP Parallelization (basic)~~**
- [ ] **Tensor Core Optimizations (In-Progress)**

----------

### Mid-Term

- [ ] **Autograd System**
- [ ] **Neural Layers**
- [ ] **Optimizers**
- [ ] **Loss Functions**
- [ ] **Training Loop**

----------

### Long-Term

- [ ] **Transformer**
- [ ] **Tokenizer**
- [ ] **Text Generation**
- [ ] **Interface**
- [ ] **Performance Benchmarks**
- [ ] **Replace OpenMP**
- [ ] **GPU Parallelization**

----------

## Current Status

> ***Early-stage Project — Tensor core under development***
>
> - *Tensor core implementation in progress*
> - *Focus on CPU performance*
> - *No neural network implemented yet*

## Contributions

*This project is currently experimental, but feedback and ideas are welcome.*

## License

> This project is under **Apache 2.0**.
