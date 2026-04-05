# ---------------------------------------------------
# Stage 1: Build the Rust Binary
# ---------------------------------------------------
FROM rust:1.80-bookworm AS builder

WORKDIR /usr/src/tvectordb

# Cache dependencies: copy manifests first, build a dummy, then copy source
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto/ proto/

# Create a dummy main so cargo can fetch and compile all dependencies
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "" > src/lib.rs && \
    echo "fn main() {}" > src/bin/benchmark.rs && \
    echo "fn main() {}" > src/bin/validate_recall.rs && \
    cargo build --release 2>/dev/null || true

# Now copy the real source code and build for real
COPY src/ src/
RUN touch src/main.rs src/lib.rs && \
    cargo build --release

# ---------------------------------------------------
# Stage 2: Create the tiny runtime image
# ---------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/tvectordb/target/release/tvectordb /usr/local/bin/tvectordb

# Create the data directory for persistent vector storage
RUN mkdir -p /app/data

# Environment variables — these are the correct defaults for Docker.
# Users can override any of these via `docker run -e TVECTORDB_DIM=384 ...`
# without losing the data directory mapping.
ENV TVECTORDB_DATA_DIR=/app/data
ENV TVECTORDB_PORT=3000
ENV TVECTORDB_GRPC_PORT=50051
ENV TVECTORDB_DIM=1536
ENV TVECTORDB_BITS=3

# Expose REST and gRPC ports
EXPOSE 3000
EXPOSE 50051

# Persistent storage volume
VOLUME ["/app/data"]

# Health check — ensures the server is actually responding
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:3000/ || exit 1

# Entrypoint — the binary reads ENV vars automatically.
# Users can still pass CLI flags to override: docker run tvectordb --dim 384 --bits 4
# The key difference: --data defaults to /app/data via ENV, not ./data
ENTRYPOINT ["tvectordb"]
