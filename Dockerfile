FROM rocm/dev-ubuntu-22.04:7.0.2

# Install rocblas (7.0.2) + build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends rocblas hipblas hipblas-dev rocblas-dev rocsolver-dev git cmake && \
    rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp (against 7.0.2 headers/libs for correct ABI)
# Pinned to tested commit for stability. To use latest:
#   replace the two lines below with: git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
ARG LLAMA_CPP_COMMIT=05fa625eac5bbdbe88b43f857156c35501421d6e
RUN git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && git checkout ${LLAMA_CPP_COMMIT} && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS="gfx906" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -- -j $(nproc) && \
    cp build/bin/* /usr/local/bin/ && \
    cp build/bin/*.so* /usr/local/lib/ 2>/dev/null; \
    ldconfig && \
    rm -rf /tmp/llama.cpp

# Patch rocBLAS AFTER compilation: ROCm 7.0.2 dropped gfx906 support.
# Replace librocblas.so runtime + TensileLibrary with 6.3 versions so that
# TRSM/SOLVE_TRI (used by SSM/Mamba models) can find gfx906 kernels.
# hipBLAS links to SONAME librocblas.so.5, so we redirect .so.5 → 6.3's .so.
RUN echo 'deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/6.3/ jammy main' \
      > /etc/apt/sources.list.d/rocm63.list && \
    apt-get update && \
    cd /tmp && apt-get download rocblas=4.3.0.60300-39~22.04 && \
    dpkg-deb -x rocblas*.deb /tmp/rocblas63 && \
    cp -f /tmp/rocblas63/opt/rocm-6.3.0/lib/librocblas.so.4.3* /opt/rocm/lib/ && \
    rm -f /opt/rocm/lib/librocblas.so.5.0.70002 && \
    ln -sf librocblas.so.4.3.60300 /opt/rocm/lib/librocblas.so.5 && \
    ln -sf librocblas.so.4.3.60300 /opt/rocm/lib/librocblas.so.4 && \
    ln -sf librocblas.so.4 /opt/rocm/lib/librocblas.so && \
    rm -rf /opt/rocm/lib/rocblas/library && \
    cp -r /tmp/rocblas63/opt/rocm-6.3.0/lib/rocblas/library /opt/rocm/lib/rocblas/library && \
    ln -sf libhipblaslt.so.1 /opt/rocm/lib/libhipblaslt.so.0 && \
    ln -sf $(readlink -f /opt/rocm/lib/libamdhip64.so) /opt/rocm/lib/libamdhip64.so.6 && \
    ldconfig && \
    rm -rf /tmp/rocblas* /etc/apt/sources.list.d/rocm63.list && \
    apt-get update && rm -rf /var/lib/apt/lists/*

ENV HSA_OVERRIDE_GFX_VERSION=9.0.6
ENV HSA_XNACK=0

WORKDIR /model
ENTRYPOINT ["llama-server"]
CMD ["--help"]
