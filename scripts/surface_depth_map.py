#!/usr/bin/env python3
"""
Script to create a grayscale map showing surface depth relative to sea level.

- Surface depth = elevation + water_depth + suspended_load - sea_level
- -1.6 feet or less → black (0)
- 0 feet or more → white (255)
- Between -1.6 and 0 → interpolate shades of gray

Usage:
    python scripts/surface_depth_map.py terrain.csv [output.png]

If output.png is not specified, it defaults to surface_depth_map.png
"""

import sys
from PIL import Image

from terrain_helpers import load_terrain_csv


# Depth thresholds for grayscale mapping
DEPTH_MIN = -1.6  # Black
DEPTH_MAX = 0.0   # White


def depth_to_grayscale(depth: float) -> int:
    """
    Convert surface depth to grayscale value (0-255).
    
    - depth <= DEPTH_MIN (-1.6): black (0)
    - depth >= DEPTH_MAX (0.0): white (255)
    - in between: linear interpolation
    """
    if depth <= DEPTH_MIN:
        return 0
    if depth >= DEPTH_MAX:
        return 255
    
    # Linear interpolation between DEPTH_MIN and DEPTH_MAX
    t = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    return int(t * 255)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/surface_depth_map.py terrain.csv [output.png]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "surface_depth_map.png"
    
    terrain = load_terrain_csv(input_path)
    print(f"Years: {terrain.years:.2f}, Sea level: {terrain.sea_level:.2f}")
    print(f"Map dimensions: {terrain.width} x {terrain.height}")
    
    # Create grayscale image
    img = Image.new('L', (terrain.width, terrain.height), color=128)
    
    for (x, y), cell in terrain.cells.items():
        depth = terrain.get_surface_depth(x, y)
        gray = depth_to_grayscale(depth)
        img.putpixel((x, y), gray)
    
    img.save(output_path)
    print(f"Output written to {output_path}")
    
    # Print some stats
    depths = [terrain.get_surface_depth(x, y) for (x, y) in terrain.cells.keys()]
    in_range = sum(1 for d in depths if DEPTH_MIN < d < DEPTH_MAX)
    below = sum(1 for d in depths if d <= DEPTH_MIN)
    above = sum(1 for d in depths if d >= DEPTH_MAX)
    
    print(f"\nDepth statistics:")
    print(f"  Cells at or below {DEPTH_MIN} (black): {below} ({100*below/len(depths):.1f}%)")
    print(f"  Cells between {DEPTH_MIN} and {DEPTH_MAX} (gray): {in_range} ({100*in_range/len(depths):.1f}%)")
    print(f"  Cells at or above {DEPTH_MAX} (white): {above} ({100*above/len(depths):.1f}%)")


if __name__ == '__main__':
    main()

