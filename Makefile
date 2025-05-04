.PHONY: setup prepare-data download-model train-local train-sagemaker evaluate clean help create-dirs

# Default target
help:
	@echo "Available commands:"
	@echo "  make create-dirs      - Create the project directory structure"
	@echo "  make setup            - Install dependencies"
	@echo "  make prepare-data     - Download and process the HealthCareMagic-100K dataset"
	@echo "  make download-model   - Download LLaMA 3.1 8B model from Hugging Face"
	@echo "  make train-local      - Run training on local machine"
	@echo "  make train-sagemaker  - Run training on SageMaker"
	@echo "  make evaluate MODEL=path/to/model [BASELINE=1] - Evaluate model (with baseline comparison if BASELINE=1)"
	@echo "  make clean            - Remove temporary files"

# Create project directory structure
create-dirs:
	mkdir -p data/raw data/processed
	mkdir -p models
	mkdir -p configs
	mkdir -p logs
	mkdir -p outputs
	mkdir -p outputs/evaluation
	mkdir -p notebooks
	mkdir -p src/{data,training,evaluation,analysis,utils}
	mkdir -p scripts
	touch src/__init__.py
	touch src/data/__init__.py
	touch src/training/__init__.py
	touch src/evaluation/__init__.py
	touch src/analysis/__init__.py
	touch src/utils/__init__.py
	@echo "Project directory structure created successfully"

# Setup environment
setup: create-dirs
	pip install -e .
	@echo "Python environment setup complete"

# Setup environment (alternative if pip install -e . doesn't work)
setup-alt: create-dirs
	@echo "Setting up Python environment without installation..."
	@echo "export PYTHONPATH=$${PYTHONPATH}:$(shell pwd)" > .env
	@echo "Run 'source .env' to add the project to your Python path"

# Data preparation
prepare-data:
	python -m src.data.processor --input_dir data/raw --output_dir data/processed --download

# Download model
download-model:
	python -m src.utils.download_model --model_name meta-llama/Meta-Llama-3.1-8B --output_dir models

# Local training
train-local:
	python -m src.training.train --config configs/train_config.json

# SageMaker training
train-sagemaker:
	python -m src.training.train --config configs/train_config.json --sagemaker

# Evaluation
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required. Usage: make evaluate MODEL=path/to/model [BASELINE=1]"; \
		exit 1; \
	fi
	@if [ "$(BASELINE)" = "1" ]; then \
		python -m src.evaluation.evaluate --model_path "$(MODEL)" --compare_baseline; \
	else \
		python -m src.evaluation.evaluate --model_path "$(MODEL)"; \
	fi

# Clean temporary files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage