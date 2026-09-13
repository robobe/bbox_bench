#!/usr/bin/env python3
"""Check that UDP metadata is bound before a one-frame request can start."""
import socket
import time

from bbox_bench.client import LocalViewer


def main():
    viewer = LocalViewer("unused", "Auto", 0)
    viewer.start_metadata_receiver()
    assert viewer.receiver_ready.is_set() and not viewer.receiver_error
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"0,frame,,,,,,,,0\n", ("127.0.0.1", viewer.metadata_port)); sock.close()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with viewer.lock:
            if 0 in viewer.metadata: break
        time.sleep(.01)
    with viewer.lock: assert viewer.metadata[0]["complete_at"] is not None
    viewer.close()


if __name__ == "__main__": main()
