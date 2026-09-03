# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""UDP state streaming between a headless MuJoCo master and a display-only slave viewer.

Split responsibility, not split simulation: the master (e.g. the Orange Pi) is the only
side that steps physics. The slave (e.g. a laptop with real desktop OpenGL) never calls
mj_step — each time a new packet arrives it overwrites qpos/qvel in its own MjData and
calls mj_forward, which recomputes everything the renderer needs (body poses, contacts,
sensor data) without integrating anything. Both sides must load the identical MJCF, so
qpos/qvel line up index-for-index. See docs/dev/sim_stream.md.

Transport is plain UDP, "latest wins" — never TCP. Every packet is a full snapshot, not
a delta, so a dropped packet just costs one skipped redraw; buffering a backlog behind a
slow/blocked receiver would be strictly worse than showing slightly stale state.

Wire format assumes both ends are little-endian (true for the ARM Linux Pi and any
x86_64/ARM laptop this pairs with) — qpos/qvel float64 arrays are sent as raw native
bytes with no byte-swapping, for speed and simplicity.
"""

import select
import socket
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

# magic, seq, nq, nv — a fixed-size header in front of the qpos/qvel payload.
# `magic` guards against pointing a receiver at an unrelated UDP source; `nq`/`nv` guard
# against master and slave having loaded different (or drifted) MJCF models.
_HEADER = struct.Struct("<IIII")
_MAGIC = 0x6D42_6F31  # "mBo1": microban state-stream, format version 1

DEFAULT_STREAM_PORT = 9761


class StateSender:
    """Master side: broadcasts (qpos, qvel) over UDP once per tick. Fire-and-forget."""

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, model: "mujoco.MjModel", data: "mujoco.MjData") -> None:
        header = _HEADER.pack(_MAGIC, self._seq, model.nq, model.nv)
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        payload = header + data.qpos.tobytes() + data.qvel.tobytes()
        try:
            self._sock.sendto(payload, self._addr)
        except OSError:
            # No listener yet, or a transient network error — display issues must
            # never affect the physics loop, which is the whole point of the split.
            pass

    def close(self) -> None:
        self._sock.close()


class StateReceiver:
    """Slave side: keeps only the newest state packet, applied on demand.

    ``poll()`` waits up to ``timeout`` seconds for the first packet, then drains any
    further backlog non-blocking — so a slow slave (or a viewer.sync() that took a
    while) always renders the latest physics state, never a queue of stale ones.
    """

    def __init__(self, host: str, port: int, timeout: float = 0.1) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._timeout = timeout
        self._warned_mismatch = False

    def poll(self, data: "mujoco.MjData") -> bool:
        """Apply the latest available packet into ``data.qpos``/``data.qvel``.

        Returns True if new state was applied.
        """
        applied = False
        payload = self._recv(self._timeout)
        while payload is not None:
            if self._apply(payload, data):
                applied = True
            payload = self._recv(0.0)
        return applied

    def _recv(self, timeout: float) -> bytes | None:
        ready, _, _ = select.select([self._sock], [], [], timeout)
        if not ready:
            return None
        try:
            payload, _ = self._sock.recvfrom(65536)
            return payload
        except OSError:
            return None

    def _apply(self, payload: bytes, data: "mujoco.MjData") -> bool:
        if len(payload) < _HEADER.size:
            return False
        magic, _seq, nq, nv = _HEADER.unpack_from(payload)
        if magic != _MAGIC:
            return False
        if nq != data.qpos.shape[0] or nv != data.qvel.shape[0]:
            if not self._warned_mismatch:
                print(
                    f"state_stream: model mismatch (packet has nq={nq}, nv={nv}; this "
                    f"MjData expects nq={data.qpos.shape[0]}, nv={data.qvel.shape[0]}) — "
                    "is the slave loading the same MJCF as the master? Dropping packets.",
                    flush=True,
                )
                self._warned_mismatch = True
            return False
        expected_len = _HEADER.size + (nq + nv) * 8
        if len(payload) != expected_len:
            return False
        offset = _HEADER.size
        data.qpos[:] = np.frombuffer(payload, dtype=np.float64, count=nq, offset=offset)
        data.qvel[:] = np.frombuffer(payload, dtype=np.float64, count=nv, offset=offset + nq * 8)
        return True

    def close(self) -> None:
        self._sock.close()


def parse_host_port(value: str, default_port: int) -> tuple[str, int]:
    """Parse a "HOST" or "HOST:PORT" CLI argument, defaulting the port if omitted."""
    if ":" in value:
        host, port_str = value.rsplit(":", 1)
        return host, int(port_str)
    return value, default_port
