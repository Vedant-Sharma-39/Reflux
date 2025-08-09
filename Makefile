# Makefile for the Reflux Simulation Framework
#
# A simple, clean wrapper around the main control script: manage.py

.PHONY: help install launch clean plot figures fig1 fig2 fig3 fig4 debug
.DEFAULT_GOAL := help

# --- Configuration ---
MANAGER := python3 manage.py

# --- Main Targets ---
help:
	@$(MANAGER) --help

install:
	@echo "Installing dependencies from requirements.txt..."
	@python3 -m pip install -r requirements.txt
	@echo "✅ Done. Activate your virtual environment if you have one."

## Launch a campaign. Can be run interactively or with an argument.
## e.g., `make launch` (interactive) or `make launch EXP=phase_diagram`
launch:
	@$(MANAGER) launch $(EXP)

## Clean a campaign's data.
## e.g., `make clean` (interactive) or `make clean EXP=phase_diagram`
clean:
	@$(MANAGER) clean $(EXP)

## Generate a plot.
## e.g., `make plot` (interactive) or `make plot FIG=fig2`
plot:
	@$(MANAGER) plot $(FIG)

## Generate all figures.
figures:
	@$(MANAGER) plot all

## Shortcuts to generate specific figures. e.g., `make fig1`
fig1 fig2 fig3 fig4:
	@$(MANAGER) plot $@

status:
	@$(MANAGER) status $(EXP)


## Debug a task from a campaign.
## e.g., `make debug` (interactive) or `make debug EXP=phase_diagram`
debug:
	@$(MANAGER) debug $(EXP)