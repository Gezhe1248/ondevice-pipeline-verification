# -*- coding: utf-8 -*-
"""Verification tooling for an on-device multi-stage inference pipeline.

Modules
    guard            deterministic post-processing layer for generated text
    offline_test     process-level network isolation test (no outbound traffic)
    verify_onnx      exported-model vs framework numerical consistency check
    bench_backends   cold-start / throughput comparison across inference backends
"""

__all__ = ["guard", "offline_test", "verify_onnx", "bench_backends"]
__version__ = "0.1.0"
