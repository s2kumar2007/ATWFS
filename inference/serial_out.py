"""Serial link for transmitting control packets to a microcontroller (Arduino).

Packet format:
  ``[angle, speed, trust, obstacle_flag]\\n``
  e.g. ``[+2.50, 0.800, 0.920, 0]\\n``
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def format_packet(
    angle: float,
    speed: float,
    trust: float,
    obstacle_flag: Union[int, bool],
) -> str:
    """Format a 4-tuple control packet into an ASCII serial string."""
    obs = 1 if obstacle_flag else 0
    return f"[{float(angle):+.2f}, {float(speed):.3f}, {float(trust):.3f}, {obs}]\n"


class SerialSender:
    """Serial transmitter for ``[angle, speed, trust, obstacle_flag]`` packets.

    Supports hardware serial via ``pyserial``, mock serial objects for testing,
    and ``--dry-run`` mode for hardware-free simulation.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        dry_run: bool = False,
        timeout: float = 0.1,
        serial_conn: Optional[Any] = None,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.dry_run = bool(dry_run)
        self.timeout = float(timeout)
        self.conn = serial_conn
        self.history: List[str] = []
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._estop = False
        self._last_send = time.monotonic()
        self._angle = 0.0
        self._speed = 0.0

        if self.conn is None and not self.dry_run:
            if not self.port:
                print("NEEDS USER DATA: Connected Arduino serial port (use --dry-run for simulation)")
                self.dry_run = True
            else:
                try:
                    import serial  # type: ignore

                    self.conn = serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout)
                    logger.info("Opened serial port %s @ %d baud", self.port, self.baud)
                except Exception as exc:
                    print(f"NEEDS USER DATA: Connected Arduino serial port (use --dry-run for simulation) [{exc}]")
                    logger.warning("Serial port %s unavailable (%s); falling back to dry-run.", self.port, exc)
                    raise OSError("Requested serial port unavailable") from exc

        self._monitor = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._monitor.start()

    def emergency_stop(self, reason: str = "requested") -> str:
        """Latch a stop until the sender is closed and explicitly restarted."""
        with self._lock:
            self._estop = True
            logger.warning("SAFETY STOP (latched): %s", reason)
            return self.send(0.0, 0.0, 0.0, 1)

    def check_watchdog(self, now: Optional[float] = None) -> None:
        with self._lock:
            elapsed = (time.monotonic() if now is None else now) - self._last_send
            if elapsed > 0.250 and not self._estop:
                self.emergency_stop("command freshness guard expired (250 ms; 300 ms safety budget)")

    def _watchdog_loop(self) -> None:
        while not self._closed.wait(0.005):
            self.check_watchdog()

    def send(self, angle: float, speed: float, trust: float, obstacle_flag: Union[int, bool]) -> str:
        with self._lock:
            self.check_watchdog()
            return self._send_locked(angle, speed, trust, obstacle_flag)

    def _send_locked(
        self,
        angle: float,
        speed: float,
        trust: float,
        obstacle_flag: Union[int, bool],
    ) -> str:
        """Format and transmit a control packet."""
        with self._lock:
            values = (float(angle), float(speed), float(trust))
            if not all(math.isfinite(v) for v in values):
                self._estop = True
            if self._estop:
                angle, speed, trust, obstacle_flag = max(self._angle-5.0, min(self._angle+5.0, 0.0)), 0.0, 0.0, 1
                self._angle, self._speed = angle, speed
            else:
                angle = max(-30.0, min(30.0, values[0]))
                angle = max(self._angle - 5.0, min(self._angle + 5.0, angle))
                trust = max(0.0, min(1.0, values[2]))
                speed = max(0.0, min(1.5, values[1]))
                if obstacle_flag:
                    speed = 0.0
                elif trust < 0.4:
                    speed = min(speed, 0.3)
                self._angle, self._speed = angle, speed
            self._last_send = time.monotonic()
        msg = format_packet(angle, speed, trust, obstacle_flag)
        self.history.append(msg)

        if self.dry_run or self.conn is None:
            print(f"SERIAL DRY-RUN: {msg.strip()}", flush=True)
            return msg

        try:
            self.conn.write(msg.encode("ascii"))
            if hasattr(self.conn, "flush"):
                self.conn.flush()
        except Exception as exc:
            logger.warning("Serial write failed (%s); switching to dry-run.", exc)
            self._estop = True
            raise OSError("Serial write failed; stop latched; firmware watchdog must stop hardware") from exc

        return msg

    def send_packet(self, packet: Union[Sequence[Any], Tuple[float, float, float, int]]) -> str:
        """Transmit a 4-element tuple/list ``(angle, speed, trust, obstacle_flag)``."""
        return self.send(float(packet[0]), float(packet[1]), float(packet[2]), int(packet[3]))

    def close(self) -> None:
        """Close the underlying serial connection if open."""
        self._closed.set()
        try:
            self.emergency_stop("shutdown")
        except OSError:
            logger.error("Shutdown stop could not reach serial port; firmware watchdog required")
        if self.conn is not None and hasattr(self.conn, "close"):
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __enter__(self) -> "SerialSender":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def main() -> None:
    """CLI entry point for testing serial output or dry-run mode."""
    ap = argparse.ArgumentParser(description="Transmit ATWFS control packets over serial.")
    ap.add_argument("--serial-port", default=None, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--dry-run", action="store_true", help="Print packets without opening hardware port")
    ap.add_argument("--test-packets", type=int, default=5, help="Number of test packets to send")
    args = ap.parse_args()

    if not args.dry_run:
        ap.error("Test packets are allowed only with --dry-run; autonomous hardware commands come from inference.live")
    with SerialSender(port=args.serial_port, baud=args.baud, dry_run=args.dry_run) as sender:
        for i in range(args.test_packets):
            angle = (i - 2) * 2.5
            speed = max(0.0, 0.8 - 0.1 * i)
            trust = max(0.2, 0.95 - 0.15 * i)
            obs = 1 if i == args.test_packets - 1 else 0
            sender.send(angle, speed, trust, obs)


if __name__ == "__main__":
    main()
