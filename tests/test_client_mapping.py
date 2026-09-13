#!/usr/bin/env python3
"""Check that a server selection is visible before local validation."""
from bbox_bench.client import relative_source


def main():
    assert relative_source("/home/radxa/gst-rknn/datasets/UAV123/car1", "/home/radxa/gst-rknn/datasets") == "UAV123/car1"


if __name__ == "__main__":
    main()
