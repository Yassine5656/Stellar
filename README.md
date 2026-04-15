# Stellar

**Stellar** is a simplified **Large Language Model** (LLM), made from scratch* and written in **C++20**.
This project focuses on deep-learning, starting with a Tensor Library with an emphasis on CPU optimization.

> ***Note:** The project currently uses OpenMP to do parallelized computations. It may change in a future version.*

## Key features

### Tensor Core
- **Memory Efficient Management:** Contiguous Row-major data structure to maximize cache use and minimize allocation overhead.
- **Hybrid Parallelism:** Leverage **OpenMP** for accelerated computations. The Tensor Core employs ```PARALLEL_THRESHOLD``` strategy to ensure a optimal performance trade-off between threading overhead and compute speed.
- **Weight Initialization:** Includes native implementations of industry-standard initialization, including **Xavier Uniform** and **He Normal**.
- **Persistence:** Local-based persistence designed for I/O efficiency during save and load.

## Architecture
```text
stellar/
├── include/core/                 # API includes (headers)
├── libstellar/core/              # LLM Core implementation
├── libstellar/CMakeLists.txt     # LLM Core Library build
├── src/                          # Application EntryPoint (CLI)
├── tests/                        # Unit testing suite
└── CMakeLists.txt                # Global Project Build
```

## Getting Started

### Prerequisites
- ***Compiler:*** GCC 10+ or Clang 10+ (C++20 required)
- ***Build System:*** CMake 3.20+
- ***Libraries:*** OpenMP

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

- [x] **Core Tensor**
- [ ] **Tensor Core Optimizations**

- [ ] **Neural Layers**
- [ ] **Automatic Differentiation**
- [ ] **Transformer Blocks**

- [ ] **LLM Interface**
- [ ] **Performance Benchmarks**