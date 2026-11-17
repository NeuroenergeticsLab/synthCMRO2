# Synthetic CMRO2: figures from the public data, and the full model pipeline.
#
# Figures and tables of the manuscript (public data, no GPU, a few minutes):
#   make install-figures   Python environment with the figure dependencies (uv)
#   make figures           Figures 2-4 and S2 and their statistics -> results/manuscript/
#   make benchmark         Table S2 -> results/manuscript/stats/noise_benchmark.csv
#   make test              unit tests (PyTorch tests are skipped without PyTorch)
#
# Models (install PyTorch for your CUDA version first, see README):
#   make install           full environment (models and figures)
#   make apply IN="CBF=cbf.nii.gz T2S=t2s.nii.gz" OUT=synth_cmro2.nii.gz   one participant, CBF&T2* model
#   make check | train MODELS="cbf_t2s" | select | predict | export TAG=...   full pipeline (config/model.yaml)
#   make figures-individual TAG=...    figures from your own export (config/figures_individual.yaml)

MODEL_CONFIG ?= config/model.yaml
FIGURES_CONFIG ?= config/figures.yaml
INDIVIDUAL_CONFIG ?= config/figures_individual.yaml
MODELS ?= reference without_cbv cbf_t2s cbf_t2 t2_t2s cbf t2s t2 t1 degree_centrality
MODEL ?= cbf_t2s
DEVICE ?= cpu
TAG ?= $(shell date +%Y%m%d)
RUN = uv run synthcmro2

.PHONY: install install-figures test lint figures benchmark apply check train select predict export pls \
	figures-individual public-tables

install-figures:
	uv venv && uv pip install -e ".[figures,dev]"

install:
	uv venv && uv pip install -e ".[model,figures,dev]"

test:
	uv run pytest

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

figures:
	$(RUN) figures --config $(FIGURES_CONFIG)

benchmark:
	$(RUN) benchmark --config $(FIGURES_CONFIG)

apply:
	$(RUN) model --config $(MODEL_CONFIG) apply --model $(MODEL) --input $(IN) --out $(OUT) --device $(DEVICE)

check:
	$(RUN) model --config $(MODEL_CONFIG) check

train:
	$(RUN) model --config $(MODEL_CONFIG) train --models $(MODELS) --device $(DEVICE)

select:
	$(RUN) model --config $(MODEL_CONFIG) select --device $(DEVICE)

predict:
	$(RUN) model --config $(MODEL_CONFIG) predict --device $(DEVICE)

export:
	$(RUN) model --config $(MODEL_CONFIG) export --tag $(TAG)
	$(RUN) model --config $(MODEL_CONFIG) export --datasets test_retest --tag $(TAG)_test_retest

pls:
	$(RUN) pls --config $(INDIVIDUAL_CONFIG) --tag $(TAG)

figures-individual:
	$(RUN) figures --config $(INDIVIDUAL_CONFIG) --tag $(TAG) --exclude-parcels TE1m TE2a TE1p TE2p a10p 11l

public-tables:
	$(RUN) public-tables --config $(INDIVIDUAL_CONFIG) --tag $(TAG) --exclude-parcels TE1m TE2a TE1p TE2p a10p 11l \
		--out data/public
