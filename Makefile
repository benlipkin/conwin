SHELL := /usr/bin/env bash
EXEC = python=3.10
PACKAGE = conwin
INSTALL = python -m pip install
ACTIVATE = source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin main

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
	@conda create -yn $(PACKAGE) $(EXEC)
	@$(ACTIVATE) ; conda install -yc conda-forge git-lfs
	@$(ACTIVATE) ; $(INSTALL) -e "."

## test      : run testing pipeline.
.PHONY : test
test : mypy pylint
mypy : env html/mypy/index.html
pylint : env html/pylint/index.html
html/mypy/index.html : **/*.py
	@$(ACTIVATE) ; mypy \
	-p $(PACKAGE) \
	--ignore-missing-imports \
	--html-report $(@D)
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : **/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(PACKAGE) \
	--disable C0114,C0115,C0116 \
	--generated-members torch.* \
	--output-format=colorized,json:$@ \
	|| pylint-exit $$?

## run	   : run the main experiment.
.PHONY : run
run : env 1024 512 256 128 64 32 16 8
1024 512 256 128 64 32 16 8 : **/*.py
	@$(ACTIVATE) ; accelerate launch runner.py \
	--window_size $@ \
	--batch_size $$(expr 16384 / $@)

## scores  : run brainscore pipeline.
.PHONY : scores
scores : analysis/src/run.py
	@source activate brainscore; cd $(<D); python $(<F)

## plots   : generate plots.
.PHONY : plots
plots : analysis/src/plots.py scores 
	@source activate brainscore; cd $(<D); python $(<F)
