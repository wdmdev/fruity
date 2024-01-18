.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation dvcfood dvcfruit trainfood

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fruity
PYTHON_VERSION = 3.10.9
PYTHON_INTERPRETER = python
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
# create_environment:
# 	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y
	$(CONDA_ACTIVATE) $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

dvcfood:
	dvc pull
	sudo apt install p7zip-full
	7z x foods_101.7z -odata/raw

dvcfruit:
	dvc pull
	sudo apt install p7zip-full
	7z x fruits_360.7z -odata/raw



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Train model
train:
	$(CONDA_ACTIVATE) $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) src/fruity/train.py


trainfood:
	$(CONDA_ACTIVATE) $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) src/fruity/train.py experiment=train_food


# Serve API
serve_api: 
	cd app/backend && uvicorn fruity_api:app --reload

#Build fastapi docker image
build_local_api_image:
	docker build -t local_fruity_api -f dockerfiles/local.api.dockerfile .

build_gc_api_image:
	docker build -t fruity_api -f dockerfiles/gc.api.dockerfile .

# Run fastapi docker image
run_api_image:
	docker run -p 80:80 fruity_api

# Check pre-commit hooks
pre-commit:
	pre-commit run --all-files

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
