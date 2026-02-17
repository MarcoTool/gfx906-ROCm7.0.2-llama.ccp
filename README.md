# llama.cpp Docker for AMD MI50 (gfx906) + ROCm 7.0.2

Docker-based llama.cpp deployment for AMD gfx906 GPUs (**Instinct MI50**, **Radeon VII**), with a critical fix for **SSM/Mamba model support** (e.g., Qwen3-Next-80B-A3B).

## The Problem

ROCm 7.0.2 dropped gfx906 support in rocBLAS. While basic GEMM operations can work with `HSA_OVERRIDE_GFX_VERSION=9.0.6` and patched TensileLibrary files, **TRSM (triangular solve)** operations — used by SSM/Mamba models' `SOLVE_TRI` — fail with:

```
rocBLAS error from hip error code: 'hipErrorInvalidDeviceFunction':98
ggml_cuda_compute_forward: SOLVE_TRI failed
```

This affects any model using SSM architecture (Qwen3-Next, Jamba, etc.) but NOT pure attention models (GLM-4.7-Flash, Qwen3-32B, etc.).

## The Fix

The Dockerfile implements a 3-layer fix:

### 1. Replace rocBLAS runtime with 6.3 version

The 7.0.2 `librocblas.so` has TRSM kernels compiled without gfx906 support. These are **embedded in the .so binary**, not in TensileLibrary files. Simply copying TensileLibrary `.co`/`.hsaco` files from 6.3 does NOT fix this — the runtime itself must be from 6.3.

### 2. SONAME redirection

hipBLAS 7.0.2 has `DT_NEEDED: librocblas.so.5`, but rocBLAS 6.3 provides `librocblas.so.4`. The fix:

```dockerfile
rm -f /opt/rocm/lib/librocblas.so.5.0.70002
ln -sf librocblas.so.4.3.60300 /opt/rocm/lib/librocblas.so.5
```

The original `.so.5.0.70002` **must be deleted** before creating the symlink, otherwise `ldconfig` will automatically restore it.

### 3. Cross-version dependency symlinks

rocBLAS 6.3 depends on `libhipblaslt.so.0` and `libamdhip64.so.6`, but ROCm 7.0.2 provides `.so.1` and `.so.7` respectively:

```dockerfile
ln -sf libhipblaslt.so.1 /opt/rocm/lib/libhipblaslt.so.0
ln -sf $(readlink -f /opt/rocm/lib/libamdhip64.so) /opt/rocm/lib/libamdhip64.so.6
```

### Critical: Build Order

**llama.cpp must be compiled BEFORE patching rocBLAS.** If you patch first, cmake links against rocBLAS 6.3's incomplete dependency chain in the 7.0.2 environment, causing the build to silently produce empty binaries.

## Hardware Tested

| Component | Spec |
|-----------|------|
| GPU | 2x AMD Instinct MI50 32GB |
| VBIOS | V420 (Vega 20 server variant) |
| Power Cap | 140W per GPU |
| GPU0 Bus | PCIe x16 Gen3 (16 GB/s) |
| GPU1 Bus | PCIe x4 via DMI 3.0 (1.25 GB/s) |
| P2P | Not available (Small BAR, different root complex) |
| System RAM | 31GB DDR4 + 16GB swap |
| Storage | 457GB NVMe SSD |
| OS | Ubuntu 22.04.5 LTS (kernel 6.8.0-94-generic) |
| Host Driver | ROCm 6.3 kernel driver |
| Container | ROCm 7.0.2 userspace |
| llama.cpp | build 1, commit [`05fa625`](https://github.com/ggml-org/llama.cpp/commit/05fa625) (2026-02-16) |

> **PCIe topology note:** GPU1 connects through the PCH's DMI 3.0 bus (1.25 GB/s), which severely penalizes Tensor Parallelism (25-41% speed loss). However, **Pipeline Parallelism** (`--split-mode layer`) only transfers ~8KB activations per token, so the DMI bottleneck has negligible impact.

## Benchmark Results

All benchmarks conducted on MI50 32GB with V420 VBIOS and 140W power cap per GPU.

### Qwen3-Next-80B-A3B-Instruct Q4_K_S — Dual GPU PP

`docker-compose.dual-gpu.yml` | GPU0: 22.5GB VRAM, GPU1: 20.6GB VRAM

| Context | Prompt Speed | Generation Speed |
|---------|-------------|-----------------|
| ~0      | 64.4 tok/s  | 25.1 tok/s      |
| ~2K     | 398.2 tok/s | 24.4 tok/s      |
| ~4K     | 412.5 tok/s | 24.0 tok/s      |
| ~8K     | 390.5 tok/s | 24.3 tok/s      |

- Tool calling: working (single-tool, multi-tool response integration)
- Generation speed stable across context lengths (SSM architecture benefit)
- Requires rocBLAS 6.3 patch for SOLVE_TRI

### GLM-4.7-Flash Q4_K_M — Single GPU

`docker-compose.single-gpu.yml` | GPU0: ~17.3GB VRAM

| Context | Prompt Speed | Generation Speed |
|---------|-------------|-----------------|
| ~0      | 127 tok/s   | 57.4 tok/s      |
| ~2K     | 703 tok/s   | 54.7 tok/s      |
| ~4K     | 557 tok/s   | 50.8 tok/s      |
| ~8K     | 394 tok/s   | 47.5 tok/s      |

- Tool calling: working (single-tool, multi-tool, tool response integration)
- No rocBLAS patch needed (pure attention model, no SOLVE_TRI)

### Comparison

| Model | Active Params | GPUs | Gen Speed | Needs rocBLAS Fix |
|-------|--------------|------|-----------|-------------------|
| GLM-4.7-Flash Q4_K_M | 3B (MoE) | 1x MI50 | 57 tok/s | No |
| Qwen3-Next-80B-A3B Q4_K_S | 3B (MoE+SSM) | 2x MI50 PP | 25 tok/s | **Yes** |

## Usage

### Build

```bash
docker compose build
```

### Run — Single GPU

For attention-only models (GLM-4.7-Flash, Qwen3, LLaMA, Mistral, etc.):

```bash
# Place your GGUF model in ./model/
cp docker-compose.single-gpu.yml docker-compose.yml
docker compose up -d
```

### Run — Dual GPU Pipeline Parallelism

For large models or SSM/Mamba models (Qwen3-Next-80B, Jamba, etc.):

```bash
cp docker-compose.dual-gpu.yml docker-compose.yml
docker compose up -d
```

### Monitor

```bash
docker compose logs -f
```

## Configuration Reference

### Single GPU (`docker-compose.single-gpu.yml`)

```yaml
environment:
  - HIP_VISIBLE_DEVICES=0          # Use GPU0 only
command:
  - --model
  - /model/your-model.gguf
  - --n-gpu-layers
  - "999"                           # Offload all layers to GPU
```

### Dual GPU PP (`docker-compose.dual-gpu.yml`)

```yaml
environment:
  - HIP_VISIBLE_DEVICES=0,1        # Use both GPUs
command:
  - --model
  - /model/your-model.gguf
  - --n-gpu-layers
  - "999"
  - --split-mode
  - layer                           # Pipeline Parallelism
  - --no-mmap                       # Avoid swap thrashing if RAM < model size
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `HIP_VISIBLE_DEVICES` | `0` for single GPU, `0,1` for dual GPU |
| `--split-mode layer` | Pipeline Parallelism (required for dual GPU with DMI bottleneck) |
| `--no-mmap` | Direct file read instead of memory-mapping (use when system RAM < model file size) |
| `--n-gpu-layers 999` | Offload all layers to GPU |
| `HSA_OVERRIDE_GFX_VERSION=9.0.6` | Tell ROCm runtime this is a gfx906 device |

## Updating llama.cpp

The Dockerfile pins llama.cpp to a tested commit for stability. To build with the latest version:

```bash
docker compose build --build-arg LLAMA_CPP_COMMIT=HEAD
```

Or edit the `ARG LLAMA_CPP_COMMIT` line in the Dockerfile to a newer commit hash.

## Compatibility

This fix applies to all **gfx906** devices:

| GPU | VRAM | Status |
|-----|------|--------|
| AMD Instinct MI50 | 32GB / 16GB | Tested |
| AMD Radeon VII | 16GB | Compatible (same gfx906 chip) |

## License

MIT License. See [LICENSE](LICENSE) for details.
