#!/usr/bin/env python3
"""
Script to extract the minimum (elevation + water_depth + suspended_load - sea_level) 
for each column (x-value) from a terrain.csv file.

Usage:
    python scripts/extract_column_min_depths.py terrain.csv [output.csv]

If output.csv is not specified, it defaults to column_min_depths.csv
"""

import csv
import sys
from collections import defaultdict

from terrain_helpers import load_terrain_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_column_min_depths.py terrain.csv [output.csv]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "column_min_depths.csv"
    
    terrain = load_terrain_csv(input_path)
    print(f"Years: {terrain.years:.2f}, Sea level: {terrain.sea_level:.2f}")
    
    column_min_depths = defaultdict(lambda: float('inf'))
    
    for (x, y), cell in terrain.cells.items():
        depth = terrain.get_surface_depth(x, y)
        if depth < column_min_depths[x]:
            column_min_depths[x] = depth
    
    # Write output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'min_depth'])
        
        for x in sorted(column_min_depths.keys()):
            writer.writerow([x, column_min_depths[x]])
    
    print(f"Output written to {output_path}")
    print(f"Columns processed: {len(column_min_depths)}")
    
    # Print some summary stats
    min_depth = min(column_min_depths.values())
    max_depth = max(column_min_depths.values())
    print(f"Min depth across all columns: {min_depth:.2f}")
    print(f"Max depth across all columns: {max_depth:.2f}")


if __name__ == '__main__':
    main()
