PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest
STREAMLIT = .venv/bin/streamlit

.PHONY: test benchmark dataset dashboard onnx clean help

help:
	@echo "Available commands:"
	@echo "  make test        - Run all 47 automated unit and integration tests"
	@echo "  make benchmark   - Run full SAHI vs Standard Resize benchmark"
	@echo "  make dataset     - Synthesize 32 high-res leaves with anti-leakage GroupKFold"
	@echo "  make dashboard   - Launch interactive Streamlit Web UI (Mobile & Desktop)"
	@echo "  make onnx        - Generate and run ONNX edge inference"
	@echo "  make clean       - Remove cached artifacts and temporary logs"

test:
	$(PYTEST) -v

benchmark:
	$(PYTHON) scripts/run_benchmark.py

dataset:
	$(PYTHON) scripts/generate_dataset.py

dashboard:
	$(STREAMLIT) run app.py

onnx:
	$(PYTHON) scripts/run_onnx_inference.py

clean:
	rm -rf .pytest_cache tests/__pycache__ src/**/__pycache__
