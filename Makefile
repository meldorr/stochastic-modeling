# Set VPY to a traffic-capable interpreter (Stage 1 needs the `traffic` lib).
VPY ?= /Users/meldor/Desktop/git/deep-traffic-generation-paper/.venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: prepare train smoke generate evaluate tb clean e1 e2 e3 e4 ablation

# --- ablation study (see REPORT.md) ---
e1:
	$(VPY) experiments/e1_smoothing.py

e2:
	$(VPY) train.py --config configs/e2_global.yaml --tag e2_global
	$(VPY) generate.py --config configs/e2_global.yaml
	$(VPY) evaluate.py --config configs/e2_global.yaml
	$(VPY) experiments/eval_gen.py --exp e2_global --generated results/e2_global/generated.npz --features xy

e3:
	$(VPY) train.py --config configs/e3_cluster.yaml --tag e3_cluster
	$(VPY) generate.py --config configs/e3_cluster.yaml
	$(VPY) evaluate.py --config configs/e3_cluster.yaml
	$(VPY) experiments/eval_gen.py --exp e3_cluster --generated results/e3_cluster/generated.npz --features xy

e4:
	$(VPY) experiments/e4_raw.py --features dyn --epochs 300
	$(VPY) experiments/eval_gen.py --exp e4_raw_dyn --generated results/e4_raw_dyn/generated.npz --features dyn
	$(VPY) experiments/e4_raw.py --features xy --epochs 300
	$(VPY) experiments/eval_gen.py --exp e4_raw_xy --generated results/e4_raw_xy/generated.npz --features xy

ablation: e1 e2 e3 e4

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
