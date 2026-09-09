.PHONY: check gate transformers bilstm hookbait cost tables figures clean

DATA ?= data
PROBS ?= probs
COSTS ?= results/costs.json
THREADS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

check:
	python -c "from mlfnd.data import build_split, assert_aligned; \
	tr, va, te = build_split('$(DATA)'); assert_aligned('$(PROBS)/probs_bundle.npz', va, te)"

gate:
	python scripts/train_gate.py --data_dir $(DATA)

transformers:
	python scripts/train_transformers.py --model muril --arm baseline
	python scripts/train_transformers.py --model xlmr --arm baseline

bilstm:
	python scripts/train_bilstm.py --stage difficulty --data_dir $(DATA)
	python scripts/train_bilstm.py --stage train --data_dir $(DATA) --n_seeds 5

hookbait:
	python scripts/train_hookbait.py --model gate
	python scripts/train_hookbait.py --model muril
	python scripts/train_hookbait.py --model xlmr
	python scripts/train_hookbait.py --bundle

cost:
	$(THREADS) python scripts/benchmark_cost.py --data_dir $(DATA) --out $(COSTS)

tables:
	python scripts/make_tables.py --probs $(PROBS) --costs $(COSTS)

figures:
	python scripts/make_figures.py --probs $(PROBS) --costs $(COSTS)

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
