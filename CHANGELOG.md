# Changelog

All notable changes to the Nexus LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-12-15

### Added
- Streaming response support for real-time token generation
- Multi-GPU inference with automatic load balancing
- Prometheus metrics endpoint for monitoring request rate, latency, and GPU utilization
- Rate limiting middleware (configurable per-endpoint)
- Health check endpoint (`/health`) for container orchestration

### Changed
- Improved tokenizer caching, reducing cold-start time by 40%
- Upgraded model loading to support GGUF and SafeTensors formats
- Optimized KV-cache management for longer context windows (up to 32k tokens)

### Fixed
- Fixed memory leak in long-running inference sessions
- Resolved race condition in concurrent request handling
- Fixed incorrect token counting for multi-byte UTF-8 characters

### Deprecated
- Direct model loading via `load_model()` in favor of `ModelRegistry.load()`

## [2.0.0] - 2024-09-01

### Added
- Initial release of Nexus LLM inference engine
- RESTful API server with OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`)
- Support for Hugging Face model loading and quantization (INT8, INT4)
- Batching engine with dynamic batch sizing for throughput optimization
- Configuration-based model management via YAML config files
- Docker support with multi-stage builds
- Comprehensive test suite with pytest and async test support
- CI/CD pipeline with GitHub Actions (lint, test, build)

### Security
- API key authentication middleware
- Input sanitization and prompt length validation
- CORS configuration for cross-origin requests
