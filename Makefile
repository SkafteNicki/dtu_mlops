## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the
## same name exists. See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *
CURRENT_DIR := $(shell cd)

#################################################################################
# COMMANDS                                                                      #
#################################################################################
docs:
	mkdocs serve --dirty

lint:
	ruff check . --fix
	ruff format .

linkcheck:
	docker build \
		https://github.com/gaurav-nelson/github-action-markdown-link-check.git#master -t linkcheck:latest
	docker run \
		--rm \
		-v $(CURRENT_DIR):/github/workspace \
		linkcheck \
		"no" "no" "/github/workspace/.github/linkcheck_config.json" "/github/workspace" "-1" "no" "master" ".md" " "
