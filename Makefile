# Nexus-7B Makefile
# 开发、训练、评估常用命令

.PHONY: help install test lint format clean train serve eval data tokenizer

# ============================================================================
# 帮助信息
# ============================================================================
help:
	@echo "Nexus-7B 开发命令:"
	@echo ""
	@echo "  安装:"
	@echo "    make install        - 安装依赖"
	@echo "    make install-dev    - 安装开发依赖"
	@echo ""
	@echo "  测试:"
	@echo "    make test           - 运行所有测试"
	@echo "    make test-int       - 运行集成测试"
	@echo "    make test-cov       - 运行测试并生成覆盖率报告"
	@echo ""
	@echo "  代码质量:"
	@echo "    make lint           - 代码检查 (flake8)"
	@echo "    make format         - 格式化代码 (black + isort)"
	@echo "    make typecheck      - 类型检查 (mypy)"
	@echo ""
	@echo "  数据准备:"
	@echo "    make data           - 准备训练数据"
	@echo "    make data-sample    - 创建示例数据集"
	@echo "    make tokenizer      - 训练分词器"
	@echo ""
	@echo "  训练:"
	@echo "    make train          - 启动训练"
	@echo "    make train-lora     - LoRA微调"
	@echo "    make train-dist     - 分布式训练"
	@echo ""
	@echo "  推理:"
	@echo "    make serve          - 启动推理服务"
	@echo "    make serve-openai   - 启动OpenAI兼容API"
	@echo ""
	@echo "  评估:"
	@echo "    make eval           - 运行模型评估"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build   - 构建Docker镜像"
	@echo "    make docker-up      - 启动所有服务"
	@echo "    make docker-down    - 停止所有服务"
	@echo ""
	@echo "  清理:"
	@echo "    make clean          - 清理临时文件"
	@echo "    make clean-all      - 清理所有生成文件"

# ============================================================================
# 安装
# ============================================================================
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[dev,docs,quantization]"

# ============================================================================
# 测试
# ============================================================================
test:
	python -m pytest tests/ -v

test-int:
	python -m pytest tests/test_integration.py -v -s

test-cov:
	python -m pytest tests/ -v --cov=src/nexus_llm --cov-report=html --cov-report=term

test-quick:
	python -m pytest tests/test_model.py tests/test_data.py -v

# ============================================================================
# 代码质量
# ============================================================================
lint:
	python -m flake8 src/nexus_llm tests examples --max-line-length=100 --exclude=__pycache__

format:
	python -m black src/nexus_llm tests examples --line-length=100
	python -m isort src/nexus_llm tests examples --profile=black

format-check:
	python -m black --check src/nexus_llm tests examples --line-length=100
	python -m isort --check src/nexus_llm tests examples --profile=black

typecheck:
	python -m mypy src/nexus_llm --ignore-missing-imports

# ============================================================================
# 数据准备
# ============================================================================
data:
	python scripts/prepare_data.py --output-dir data/prepared --max-samples 100000

data-sample:
	python scripts/prepare_data.py --sample --output-dir data/sample --sample-count 1000

tokenizer:
	python scripts/train_tokenizer.py --sample --output-dir tokenizer --vocab-size 32000

tokenizer-custom:
	python scripts/train_tokenizer.py --input data/corpus.txt --output-dir tokenizer --vocab-size 32000

# ============================================================================
# 训练
# ============================================================================
train:
	python -m nexus_llm.cli train \
		--config configs/small_model_config.yaml \
		--data-dir data/prepared \
		--output-dir checkpoints

train-lora:
	python -m nexus_llm.cli train \
		--config configs/small_model_config.yaml \
		--lora \
		--lora-r 64 \
		--output-dir checkpoints/lora

train-dist:
	torchrun --nproc_per_node=4 -m nexus_llm.cli train \
		--config configs/small_model_config.yaml \
		--deepspeed configs/deepspeed/zero3.json \
		--output-dir checkpoints

train-rlhf:
	python -m nexus_llm.cli train \
		--rlhf \
		--config configs/training_config.yaml \
		--preference-data data/preferences.jsonl \
		--output-dir checkpoints/rlhf

# ============================================================================
# 推理服务
# ============================================================================
serve:
	python -m nexus_llm.cli serve \
		--model-path checkpoints/best \
		--tokenizer-path tokenizer/nexus_tokenizer.model \
		--port 8000 \
		--precision bf16

serve-openai:
	python -c "from nexus_llm import run_openai_server; run_openai_server('checkpoints/best', 'tokenizer/', '0.0.0.0', 8000)"

serve-quantized:
	python -m nexus_llm.cli serve \
		--model-path checkpoints/best \
		--tokenizer-path tokenizer/nexus_tokenizer.model \
		--precision int8 \
		--port 8000

# ============================================================================
# 评估
# ============================================================================
eval:
	python scripts/evaluate_model.py \
		--model-path checkpoints/best \
		--eval-data data/prepared/test.jsonl \
		--output-dir eval_results

eval-perplexity:
	python scripts/evaluate_model.py \
		--perplexity-samples 5000 \
		--output-dir eval_results

eval-benchmark:
	python scripts/evaluate_model.py \
		--benchmark-iterations 1000 \
		--output-dir eval_results

# ============================================================================
# Docker
# ============================================================================
docker-build:
	docker build -t nexus-llm:latest -f Dockerfile .

docker-build-training:
	docker build -t nexus-llm:training -f Dockerfile --target training .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f nexus-server

docker-train:
	docker-compose --profile training up nexus-trainer

# ============================================================================
# 清理
# ============================================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true

clean-all:
	make clean
	rm -rf data/prepared data/sample 2>/dev/null || true
	rm -rf checkpoints outputs logs 2>/dev/null || true
	rm -rf eval_results tokenizer 2>/dev/null || true
	rm -rf dist build *.egg-info 2>/dev/null || true

# ============================================================================
# Git
# ============================================================================
git-status:
	git status

git-push:
	git add . && git commit -m "update" && git push

# ============================================================================
# 快速开发流程
# ============================================================================
dev-flow: format lint test
	@echo "开发流程完成: 格式化 -> 检查 -> 测试"

quick-start: install-dev data-sample tokenizer
	@echo "快速启动完成: 安装 -> 示例数据 -> 分词器"

full-test: format lint test-cov typecheck
	@echo "完整测试完成: 格式化 -> 检查 -> 测试(覆盖率) -> 类型检查"