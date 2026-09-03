.PHONY: sync setup run stop shutdown voltage imu sim viewer sim-master sim-viewer gamepad-headless-enable gamepad-headless-disable

HOST ?= microban
ID ?=
SLAVE ?=
PORT ?= 9761

sync:
	rsync -avz \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='cad' \
		--exclude='docs' \
		--exclude='logs' \
		--exclude='src/debug' \
		--exclude='src/sim' \
		--exclude='src/model/mjcf' \
		./ $(HOST):microban

setup: sync
	ssh $(HOST) "bash -l -c 'cd microban && uv sync --frozen'"

sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50

viewer:
	PYTHONPATH=src uv run src/sim/viewer_main.py --hz 25

# Master/slave display split (see docs/dev/sim_stream.md): physics headless on this
# machine (e.g. the Pi), broadcast to `make sim-viewer` running on SLAVE (e.g. your
# laptop, over its own real GPU) instead of opening a local viewer window.
sim-master:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50 --stream-to $(SLAVE)

# Real 3D display-only viewer for a remote sim-master. PORT must match the one
# sim-master's SLAVE address uses (default 9761).
sim-viewer:
	PYTHONPATH=src uv run --group sim src/sim/sim_viewer_client.py --listen 0.0.0.0:$(PORT)

run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/main.py'"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/stop.py'"

voltage: sync
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/imu.py'"

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); B stops it. See docs/usage.md.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
