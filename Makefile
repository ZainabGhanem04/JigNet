# One command per step. Override configs/values like:  make train ARGS="--set train.epochs=2"
PY ?= python
ARGS ?=

generate:        ; $(PY) -m data.generate --config configs/generate_dataset.yaml $(ARGS)
prepare:         ; $(PY) -m data.prepare  --config configs/prepare_dataset.yaml $(ARGS)
train:           ; $(PY) train.py --config configs/train.yaml $(ARGS)
eval-pairs:      ; $(PY) eval.py --config configs/eval_pairs_test.yaml $(ARGS)
eval-network:    ; $(PY) eval.py --config configs/eval_network_test.yaml $(ARGS) && $(PY) eval.py --config configs/eval_network_coco.yaml $(ARGS)
eval-pipeline:   ; $(PY) eval.py --config configs/eval_pipeline_test.yaml $(ARGS) && $(PY) eval.py --config configs/eval_pipeline_coco.yaml $(ARGS)
eval-all: eval-pairs eval-network eval-pipeline
app:             ; $(PY) app.py
test:            ; $(PY) -m pytest -q

.PHONY: generate prepare train eval-pairs eval-network eval-pipeline eval-all app test
