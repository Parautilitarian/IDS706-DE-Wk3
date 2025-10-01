# Makefile for IDS706-eCommerce-Customer-Behavior
PYTHON = python3
BLACK = black
FLAKE8 = flake8
PYTEST = pytest

SRC = code.py test_code.py

# Flake8 settings: adjust ignore list and line length as needed
FLAKE8_IGNORE = E203,W503
FLAKE8_MAX_LINE = 88

.PHONY: all deps format lint check run test clean

# Default: format, lint, run tests
all: check test

## Install dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install black flake8 pytest

## Format all source files with black
format:
	@echo "=== Running black formatting ==="
	$(BLACK) $(SRC)

## Lint code with flake8
lint:
	@echo "=== Running flake8 linting ==="
	$(FLAKE8) --ignore=$(FLAKE8_IGNORE) --max-line-length=$(FLAKE8_MAX_LINE) $(SRC)

## Run both format and lint
check: format lint

## Run main program
run:
	@echo "=== Running main workflow ==="
	$(PYTHON) code.py

## Run unit tests
test:
	@echo "=== Running tests ==="
	$(PYTEST) -v test_code.py

## Clean up __pycache__ directories
clean:	
	@echo "=== Cleaning up __pycache__ directories ==="
	find . -type d -name "__pycache__" -exec rm -rf {} +	

# Makefile for IDS706-eCommerce-Customer-Behavior
PYTHON = python3
BLACK = black
FLAKE8 = flake8
PYTEST = pytest

SRC = code.py test_code.py

# Flake8 settings: adjust ignore list and line length as needed
FLAKE8_IGNORE = E203,W503
FLAKE8_MAX_LINE = 88

.PHONY: all deps format lint check run test clean

# Default: format, lint, run tests
all: check test

## Install dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install black flake8 pytest

## Format all source files with black
format:
	@echo "=== Running black formatting ==="
	$(BLACK) $(SRC)

## Lint code with flake8
lint:
	@echo "=== Running flake8 linting ==="
	$(FLAKE8) --ignore=$(FLAKE8_IGNORE) --max-line-length=$(FLAKE8_MAX_LINE) $(SRC)

## Run both format and lint
check: format lint

## Run main program
run:
	@echo "=== Running main workflow ==="
	$(PYTHON) code.py

## Run unit tests
test:
	@echo "=== Running tests ==="
	$(PYTEST) -v test_code.py




