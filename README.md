# llama.cpp Docker for AMD MI50 (gfx906) + ROCm 7.0.2

Docker-based llama.cpp deployment for AMD Instinct MI50 (gfx906) GPUs, with a critical fix for **SSM/Mamba model support** (e.g., Qwen3-Next-80B-A3B).

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

- 2x AMD Instinct MI50 32GB (V420 VBIOS, 140W power cap)
- GPU0: PCIe x16 Gen3, GPU1: PCIe x4 via DMI 3.0
- Pipeline Parallelism (PP) with `--split-mode layer`
- ROCm 6.3 kernel driver, ROCm 7.0.2 Docker userspace

## Benchmark Results

### Qwen3-Next-80B-A3B-Instruct Q4_K_S (dual GPU PP)

| Context | Prompt Speed | Generation Speed |
|---------|-------------|-----------------|
| ~0      | 64.4 tok/s  | 25.1 tok/s      |
| ~2K     | 398.2 tok/s | 24.4 tok/s      |
| ~4K     | 412.5 tok/s | 24.0 tok/s      |
| ~8K     | 390.5 tok/s | 24.3 tok/s      |

- Tool calling: working (single-tool, multi-tool response integration)
- Generation speed stable across context lengths (SSM architecture benefit)
- VRAM: GPU0 ~22.5GB, GPU1 ~20.6GB

### GLM-4.7-Flash Q4_K_M (single GPU, no rocBLAS patch needed)

| Context | Prompt Speed | Generation Speed |
|---------|-------------|-----------------|
| ~0      | 127 tok/s   | 57.4 tok/s      |
| ~2K     | 703 tok/s   | 54.7 tok/s      |
| ~4K     | 557 tok/s   | 50.8 tok/s      |
| ~8K     | 394 tok/s   | 47.5 tok/s      |

## Usage

### Quick Start

```bash
# Place your GGUF model in ./model/
docker compose up -d

# Check logs
docker compose logs -f
```

### docker-compose.yml Configuration

Edit `docker-compose.yml` to change the model, GPU assignment, and parameters:

```yaml
environment:
  - HIP_VISIBLE_DEVICES=0,1  # 0 for single GPU, 0,1 for dual GPU PP
command:
  - --model
  - /model/your-model.gguf
  - --n-gpu-layers
  - "999"
  - --split-mode
  - layer        # Pipeline Parallelism for dual GPU
  - --no-mmap    # Recommended when system RAM < model size
```

### Notes

- `--no-mmap` is recommended when system RAM is less than the model file size, to avoid swap thrashing during loading
- For SSM models (Qwen3-Next, etc.), the rocBLAS 6.3 patch is required
- For pure attention models (GLM, Qwen3, LLaMA, etc.), only the TensileLibrary patch is needed (but using the full fix is harmless)

## What Doesn't Work

These approaches were tried and **do NOT fix** the SOLVE_TRI issue:

| Approach | Why it fails |
|----------|-------------|
| Copy only `*gfx906*` TensileLibrary files from 6.3 | 7.0.2's runtime can't use 6.3's Tensile format |
| Replace entire TensileLibrary directory (with .dat index) | TRSM kernels are embedded in librocblas.so, not in TensileLibrary |
| `--override-tensor ssm_=CPU` | Only moves weight tensors, compute graph still executes SOLVE_TRI on GPU |
| Patch rocBLAS before compiling llama.cpp | 6.3's dependency chain is incomplete in 7.0.2 env, build produces empty binaries |
| Create .so.5 symlink without deleting .so.5.0.70002 | `ldconfig` overwrites the symlink back to 7.0.2 |

## License

The Dockerfile and docker-compose.yml are provided as-is. llama.cpp is licensed under MIT.
