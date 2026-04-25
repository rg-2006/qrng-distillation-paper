#!/bin/bash
# Generate gRPC Python stubs from proto file
# Run this once after cloning, or whenever you modify the .proto file.

set -e

cd "$(dirname "$0")/.."

mkdir -p src/proto_generated

python -m grpc_tools.protoc \
    -I proto \
    --python_out=src/proto_generated \
    --grpc_python_out=src/proto_generated \
    proto/entropy_service.proto

# Make src/proto_generated a Python package
touch src/proto_generated/__init__.py

# Fix import path in generated file (grpc_tools generates broken imports)
sed -i 's/^import entropy_service_pb2/from src.proto_generated import entropy_service_pb2/' \
    src/proto_generated/entropy_service_pb2_grpc.py 2>/dev/null || \
sed -i '' 's/^import entropy_service_pb2/from src.proto_generated import entropy_service_pb2/' \
    src/proto_generated/entropy_service_pb2_grpc.py 2>/dev/null || true

echo "✅ gRPC stubs generated in src/proto_generated/"
