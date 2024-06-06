## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the
## same name exists. See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# COMMANDS                                                                      #
#################################################################################
docs:
	mkdocs serve --dirty

lint:
	ruff check . --fix
	ruff format .
