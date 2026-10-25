"""Tests for Arduino serial transmitter and hardware safety clamp sketch."""
from pathlib import Path
import subprocess
import sys
from typing import List
import pytest

from inference.serial_out import SerialSender, format_packet


class MockSerialPort:
    """In-memory mock serial port for testing SerialSender."""

    def __init__(self) -> None:
        self.written: List[bytes] = []
        self.flushed: int = 0
        self.closed: bool = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True


def test_format_packet() -> None:
    pkt = format_packet(2.5, 0.8, 0.925, 0)
    assert pkt == "[+2.50, 0.800, 0.925, 0]\n"

    pkt_neg = format_packet(-12.345, 0.0, 0.35, True)
    assert pkt_neg == "[-12.35, 0.000, 0.350, 1]\n"


def test_serial_sender_with_mock_port() -> None:
    mock_conn = MockSerialPort()
    with SerialSender(serial_conn=mock_conn) as sender:
        msg1 = sender.send(3.25, 0.75, 0.91, 0)
        msg2 = sender.send_packet((-4.5, 0.0, 0.42, 1))

    assert len(mock_conn.written) == 2
    assert mock_conn.written[0].decode("ascii") == msg1
    assert mock_conn.written[1].decode("ascii") == msg2
    assert mock_conn.flushed == 2
    assert mock_conn.closed is True


def test_serial_sender_dry_run_and_missing_port(capsys: pytest.CaptureFixture[str]) -> None:
    # Dry run mode should print SERIAL DRY-RUN
    with SerialSender(dry_run=True) as sender:
        sender.send(1.0, 0.5, 0.8, 0)
    out_dry = capsys.readouterr().out
    assert "SERIAL DRY-RUN: [+1.00, 0.500, 0.800, 0]" in out_dry

    # Missing port without dry_run should print NEEDS USER DATA and fall back cleanly
    with SerialSender(port=None, dry_run=False) as sender2:
        sender2.send(0.0, 0.5, 0.9, 0)
    out_missing = capsys.readouterr().out
    assert "NEEDS USER DATA: Connected Arduino serial port" in out_missing


def test_serial_cli_dry_run() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "inference.serial_out", "--dry-run", "--test-packets", "3"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    lines = [line for line in proc.stdout.splitlines() if "SERIAL DRY-RUN:" in line]
    assert len(lines) == 3


def test_arduino_sketch_exists_and_has_safety_clamps() -> None:
    sketch_path = Path("arduino/safety_clamp/safety_clamp.ino")
    assert sketch_path.is_file(), "Arduino sketch must exist at arduino/safety_clamp/safety_clamp.ino"
    content = sketch_path.read_text(encoding="utf-8")
    assert "MAX_DELTA_ANGLE_DEG" in content
    assert "TRUST_MIN" in content
    assert "maxAngleForSpeed" in content
    assert "applySafetyClamp" in content
