.PHONY: sync setup run stop shutdown voltage imu zubr-imu sim viewer sim-master sim-viewer hw-stream gamepad-headless-enable gamepad-headless-disable

HOST ?= microban
ID ?=
SLAVE ?=
PORT ?= 9761
JOYSTICK ?=

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

# vendor/bam holds the SKS2401-capable subset of the bam package (see
# vendor/bam/README.md) — it's on PYTHONPATH directly, not a uv dependency.
sim:
	PYTHONPATH=src:vendor/bam uv run --group sim src/sim/sim_main.py --hz 50

viewer:
	PYTHONPATH=src uv run src/sim/viewer_main.py --hz 25

# Master/slave display split (see docs/dev/sim_stream.md): physics headless on this
# machine (e.g. the Pi), broadcast to `make sim-viewer` running on SLAVE (e.g. your
# laptop, over its own real GPU) instead of opening a local viewer window.
# JOYSTICK=1 drives vx/vy/vtheta from the STM32/zubr board's own remote-control
# joysticks instead of typed keyboard velocity steps — see docs/dev/hw_stream.md.
sim-master:
	PYTHONPATH=src:vendor/bam uv run --group sim src/sim/sim_main.py --hz 50 --stream-to $(SLAVE) $(if $(JOYSTICK),--joystick,)

# Real 3D display-only viewer for a remote sim-master. PORT must match the one
# sim-master's SLAVE address uses (default 9761).
sim-viewer:
	PYTHONPATH=src uv run --group sim src/sim/sim_viewer_client.py --listen 0.0.0.0:$(PORT)

# Streams the real robot's actuator positions (over the STM32 link, /dev/ttyS2) to a
# remote sim_viewer_client.py, for live visualization instead of a physics stream. Run
# directly on the Pi (this Makefile lives in ~/microban there too); PORT must match
# `make sim-viewer` on SLAVE. See docs/dev/hw_stream.md.
hw-stream:
	PYTHONPATH=src:vendor/bam uv run --group sim src/hw_state_stream.py --stream-to $(SLAVE)

run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/main.py'"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/stop.py'"

# NOTE: targets the old rustypot/XL330 bus (/dev/ttyAMA0), now dead hardware on this
# robot — see docs/dev/hw_stream.md's "voltage" section. Left as-is pending hardware
# documentation for whether the STM32/zubr board exposes voltage at all.
voltage: sync
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

# Separate I2C BMI088 (see docs/dev/hw_stream.md) — not the STM32/zubr board's onboard
# BHI260; use `make zubr-imu` for that one.
imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/imu.py'"

# STM32/zubr board's onboard BHI260, over /dev/ttyS2 — see docs/dev/hw_stream.md.
zubr-imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/zubr_imu.py'"

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); B stops it. See docs/usage.md.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
