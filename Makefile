# .Makefile

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

ENV_NAME ?= env-delusions
VENV_DIR ?= $(ENV_NAME)
LFS_MAX_FILE_GIB ?= 1
LFS_MAX_TOTAL_GIB ?= 10
REGULAR_MAX_FILE_MIB ?= 50

PYTHON_VENV := $(VENV_DIR)/bin/python
PYTHON_VENV_ABS := $(abspath $(PYTHON_VENV))
PIP_VENV := $(VENV_DIR)/bin/pip
CONDA_RUN := conda run -n $(ENV_NAME)
NPM_INSTALL_CMD := if [ -f package-lock.json ]; then npm ci; else npm install; fi
JSCPD_BIN := ./node_modules/.bin/jscpd
ENSURE_JSCPD := [ -x $(JSCPD_BIN) ] || $(NPM_INSTALL_CMD);

.PHONY: init init-js init-venv init-venv-viewer init-viewer init-conda \
	setup-venv setup-conda clean pyfmt pylint jslint mdfmt \
	clean-transcripts viewer check-sizes get-annotations

REMOTE_USER ?= jared_jaredmoore_org
REMOTE_HOST ?= 35.208.218.204
ANNOTATIONS_BASE ?= /opt/llm-delusions/manual_annotation_labels
ANNOTATIONS_PATH ?=

init: init-venv init-js

init-js:
	@$(NPM_INSTALL_CMD)

init-venv:
	python3 -m venv $(VENV_DIR)
	$(PIP_VENV) install -r requirements.txt
	$(PIP_VENV) install --editable .
	@echo venv > .env-type
	$(MAKE) setup-venv

init-venv-viewer:
	python3 -m venv $(VENV_DIR)
	$(PIP_VENV) install -r requirements/viewer.txt
	$(PIP_VENV) install --editable . --no-deps
	@echo venv > .env-type

init-viewer: init-venv-viewer init-js

init-conda:
	conda env create --file environment.yml
	@echo conda > .env-type
	$(CONDA_RUN) pip install --editable .
	$(MAKE) setup-conda

setup-venv:
	$(PYTHON_VENV) -m spacy download en_core_web_lg

setup-conda:
	$(CONDA_RUN) python -m spacy download en_core_web_lg

clean:
	rm -rf $(VENV_DIR) .env-type

# Code formatting: isort (imports) + black (code style)
pyfmt:
	$(PYTHON_VENV) -m isort --profile black scripts src analysis
	$(PYTHON_VENV) -m black scripts src analysis

# Static analysis: pylint
pylint:
	@pylint_status=0; \
	$(PYTHON_VENV) -m pylint scripts src analysis || pylint_status=$$?; \
	$(PYTHON_VENV) -m vulture scripts src analysis; \
	exit $$pylint_status

jslint:
	# Ensure root devDependencies (eslint, plugins) are installed
	@lint_status=0; \
	if [ ! -x ./node_modules/.bin/eslint ] || [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi; \
	./node_modules/.bin/prettier --cache --write \
		'analysis/**/*.js' \
		'analysis/**/*.css' \
		'analysis/**/*.html' \
		'analysis/**/*.json' \
		'!analysis/data/**/*' \
		'!analysis/figures/**/*' || lint_status=$$?; \
	./node_modules/.bin/eslint --cache --fix analysis || lint_status=$$?; \
	$(MAKE) jscpd-js; \
	exit $$lint_status

mdfmt:
	# Ensure Prettier is available
	@fmt_status=0; \
	if [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi; \
	./node_modules/.bin/prettier --write '**/*.md' || fmt_status=$$?; \
	$(MAKE) jscpd-md; \
	exit $$fmt_status

jscpd:
	# Ensure JS dependencies for jscpd are installed
	@$(ENSURE_JSCPD)
	$(JSCPD_BIN) --gitignore --pattern '**/*.{py,js,mjs,cjs,md,html,css,json,yml,yaml}' --reporters consoleFull

jscpd-py:
	# Ensure JS dependencies for jscpd are installed
	@$(ENSURE_JSCPD)
	$(JSCPD_BIN) --gitignore --pattern '**/*.py' --reporters consoleFull

jscpd-js:
	# Ensure JS dependencies for jscpd are installed
	@$(ENSURE_JSCPD)
	$(JSCPD_BIN) --gitignore --pattern '**/*.{js,mjs,cjs}' --reporters consoleFull

jscpd-md:
	# Ensure JS dependencies for jscpd are installed
	@$(ENSURE_JSCPD)
	$(JSCPD_BIN) --gitignore --noSymlinks --pattern '**/*.md' --reporters consoleFull

clean-transcripts:
	@if [ -d transcripts_de_ided ]; then \
		find transcripts_de_ided -type f \( \
			-name '*.html' \
		\) -print -delete; \
	else \
		echo "transcripts_de_ided does not exist"; \
	fi

viewer:
	# Ensure JS dependencies for the viewer are installed
	@if [ ! -d ./node_modules ]; then \
		echo "Installing JS dependencies..."; \
		$(NPM_INSTALL_CMD); \
	fi
	@echo "Starting local HTTP server for classification viewer on http://localhost:8000 ..."
	@echo "Press Ctrl+C to stop."
	@bash -c ' \
    if [ -f .env-type ] && grep -q "^conda$$" .env-type; then \
      RUN_CMD="$(CONDA_RUN) python analysis/viewer/viewer_server.py"; \
	elif [ -x "$(PYTHON_VENV)" ]; then \
	  RUN_CMD="$(PYTHON_VENV) analysis/viewer/viewer_server.py"; \
	else \
	  RUN_CMD="python3 analysis/viewer/viewer_server.py"; \
	fi; \
	$$RUN_CMD --directory . --port 8000 & \
	SERVER_PID=$$!; \
	trap "kill $$SERVER_PID" INT TERM EXIT; \
	sleep 1; \
	open http://localhost:8000/analysis/viewer/classification_viewer.html || true; \
	wait $$SERVER_PID'

check-sizes:
	@echo "Running repository size checks (Git LFS and regular files)..."
	@LFS_MAX_FILE_GIB=$(LFS_MAX_FILE_GIB) \
	LFS_MAX_TOTAL_GIB=$(LFS_MAX_TOTAL_GIB) \
	REGULAR_MAX_FILE_MIB=$(REGULAR_MAX_FILE_MIB) \
	bash scripts/check_repo_sizes.sh

get-annotations:
	@if [ -z "$(ANNOTATIONS_PATH)" ]; then \
		mkdir -p manual_annotation_labels; \
		rsync -rvl --ignore-existing --no-perms --no-owner --no-group --no-times --omit-dir-times \
			$(REMOTE_USER)@$(REMOTE_HOST):$(ANNOTATIONS_BASE)/ manual_annotation_labels/; \
	else \
		mkdir -p manual_annotation_labels/$(dir $(ANNOTATIONS_PATH)); \
		rsync -rvl --ignore-existing --no-perms --no-owner --no-group --no-times --omit-dir-times \
			$(REMOTE_USER)@$(REMOTE_HOST):$(ANNOTATIONS_BASE)/$(ANNOTATIONS_PATH) \
			manual_annotation_labels/$(ANNOTATIONS_PATH); \
	fi
