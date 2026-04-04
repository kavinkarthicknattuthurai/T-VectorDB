# ---------------------------------------------------
# Stage 1: Build the Rust Binary
# ---------------------------------------------------
FROM rust:1.80-slim-bookworm AS builder

# Install required system dependencies
# protobuf-compiler is required for tonic-build to compile tvectordb.proto
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/tvectordb

# Copy the entire project 
# (.dockerignore will strip out /target to avoid huge uploads)
COPY . .

# Build the release binary. This takes a few minutes but generates a hyper-optimized executable.
RUN cargo build --release

# ---------------------------------------------------
# Stage 2: Create the tiny runtime image
# ---------------------------------------------------
FROM debian:bookworm-slim

# Install any runtime dependencies (like SSL certs if needed later for HTTPS)
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Set up the working directory inside the container
WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/tvectordb/target/release/tvectordb /usr/local/bin/tvectordb

# Create the data directory for Vector Persistence
RUN mkdir -p /app/data

# Expose the REST and gRPC ports
EXPOSE 3000
EXPOSE 50051

# Environment variables for configuration default
ENV PORT=3000
ENV GRPC_PORT=50051
ENV DIMENSION=1536
ENV BIT_WIDTH=3

# Run the database!
# The CMD arguments will default to the ENV variables above, but can be overridden 
# via docker run arguments (e.g. `docker run ... --dim 384 --bits 4`)
CMD tvectordb --port ${PORT} --grpc-port ${GRPC_PORT} --dim ${DIMENSION} --bits ${BIT_WIDTH} --data /app/data
