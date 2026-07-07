# Set VPY to a traffic-capable interpreter (Stage 1 needs the `traffic` lib).
VPY ?= /Users/meldor/Desktop/git/deep-traffic-generation-paper/.venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: prepare train smoke generate evaluate tb clean

prepare:
	$(VPY) -m src.data.prepare --config configs/config.yaml

train:
	$(VPY) train.py

smoke:
	$(VPY) train.py --epochs 1 --tag smoke
	$(VPY) generate.py --n 256
	$(VPY) evaluate.py

generate:
	$(VPY) generate.py

evaluate:
	$(VPY) evaluate.py

tb:
	$(VPY) -m tensorboard.main --logdir runs

clean:
	rm -rf results/*.png results/*.json results/*.txt results/*.parquet results/*.npz runs/* checkpoints/*.pt
