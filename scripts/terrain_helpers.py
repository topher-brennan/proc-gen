"""
Shared helpers for loading terrain.csv files.
"""

import csv
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple


def u32_to_f32(bits: int) -> float:
    """Convert u32 bits to f32 (bitcast)."""
    return struct.unpack('f', struct.pack('I', bits))[0]


def u64_to_f64(bits: int) -> float:
    """Convert u64 bits to f64 (bitcast)."""
    return struct.unpack('d', struct.pack('Q', bits))[0]


@dataclass
class HexCell:
    x: int
    y: int
    elevation: float
    water_depth: float
    suspended_load: float
    rainfall: float
    erosion_multiplier: float
    uplift: float


@dataclass
class TerrainData:
    cells: Dict[Tuple[int, int], HexCell]  # (x, y) -> HexCell
    years: float
    sea_level: float
    width: int
    height: int
    seed: int
    step: int
    
    def get_surface_depth(self, x: int, y: int) -> float:
        """Get (elevation + water_depth + suspended_load - sea_level) for a cell."""
        cell = self.cells.get((x, y))
        if cell is None:
            return float('inf')
        return cell.elevation + cell.water_depth + cell.suspended_load - self.sea_level


def load_terrain_csv(path: str) -> TerrainData:
    """
    Load a terrain.csv file and return structured terrain data.
    
    CSV format:
    - Row 0: Header (consumed by csv.reader)
    - Row 1: Metadata (seed, step, years_bits, initial_max_bits, ...)
    - Row 2: Blank separator
    - Row 3: Hex data header
    - Row 4+: Hex data (x, y, elevation_bits, water_depth_bits, suspended_load_bits, 
                        rainfall_bits, erosion_multiplier_bits, uplift_bits)
    """
    cells: Dict[Tuple[int, int], HexCell] = {}
    max_x = 0
    max_y = 0
    
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        
        # Skip header row
        next(reader)
        
        # Row 1: metadata values
        metadata_row = next(reader)
        seed = int(metadata_row[0])
        step = int(metadata_row[1])
        years_bits = int(metadata_row[2])
        years = u64_to_f64(years_bits)
        sea_level = years * 0.02
        
        # Row 2: blank separator
        next(reader)
        
        # Row 3: hex data header
        next(reader)
        
        # Row 4+: hex data
        for row in reader:
            if len(row) < 8:
                continue
            
            x = int(row[0])
            y = int(row[1])
            
            elevation = u32_to_f32(int(row[2]))
            water_depth = u32_to_f32(int(row[3]))
            suspended_load = u32_to_f32(int(row[4]))
            rainfall = u32_to_f32(int(row[5]))
            erosion_multiplier = u32_to_f32(int(row[6]))
            uplift = u32_to_f32(int(row[7]))
            
            cells[(x, y)] = HexCell(
                x=x,
                y=y,
                elevation=elevation,
                water_depth=water_depth,
                suspended_load=suspended_load,
                rainfall=rainfall,
                erosion_multiplier=erosion_multiplier,
                uplift=uplift,
            )
            
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    return TerrainData(
        cells=cells,
        years=years,
        sea_level=sea_level,
        width=max_x + 1,
        height=max_y + 1,
        seed=seed,
        step=step,
    )

