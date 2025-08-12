use rand::Rng;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use rayon::prelude::*;
use image::{RgbImage};
mod gpu_simulation;
use gpu_simulation::{GpuSimulation, HexGpu};
use pollster;
mod constants;
use constants::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use noise::{NoiseFn, Perlin};

#[derive(Copy, Clone)]
struct FloodItem {
    surf: f32,
    x: usize,
    y: usize,
}

impl Eq for FloodItem {}
impl PartialEq for FloodItem { fn eq(&self, o:&Self)->bool { self.surf == o.surf } }

// Reverse ordering so BinaryHeap becomes min-heap on surface elevation
impl Ord for FloodItem {
    fn cmp(&self, other:&Self) -> Ordering {
        other.surf.total_cmp(&self.surf)
    }
}
impl PartialOrd for FloodItem { fn partial_cmp(&self, o:&Self)->Option<Ordering>{ Some(self.cmp(o)) } }

/// Fill all closed basins so initial free-water surface can drain to sea level.
fn prefill_basins(hex_map: &mut Vec<Vec<Hex>>) {
    let height = hex_map.len();
    let width  = hex_map[0].len();

    let mut in_queue = vec![vec![false; width]; height];
    let mut pq: BinaryHeap<FloodItem> = BinaryHeap::new();

    // Seed with ocean-connected cells (west edge or already below sea level)
    for y in 0..height {
        for x in 0..width {
            if x == 0 || hex_map[y][x].elevation < SEA_LEVEL {
                in_queue[y][x] = true;
                pq.push(FloodItem { surf: hex_map[y][x].elevation + hex_map[y][x].water_depth, x, y });
            }
        }
    }

    while let Some(FloodItem { surf, x, y }) = pq.pop() {
        let offsets = if (x & 1) == 0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
        for &(dx, dy) in offsets {
            let nx = x as i32 + dx as i32;
            let ny = y as i32 + dy as i32;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 { continue; }
            let (nxu, nyu) = (nx as usize, ny as usize);
            if in_queue[nyu][nxu] { continue; }

            let cell = &mut hex_map[nyu][nxu];
            let elev = cell.elevation;
            let mut newsurf = elev + cell.water_depth;
            if newsurf < surf {
                // Basin cell: raise its water until it spills
                cell.water_depth = surf - elev;
                newsurf = surf;
            }
            in_queue[nyu][nxu] = true;
            pq.push(FloodItem { surf: newsurf, x: nxu, y: nyu });
        }
    }
}

// Helper macro to detect NaN / Inf as early as possible and crash with context.
// I've been using step -1 to indicate that the value is not associated with a step,
// or the step is not known.
macro_rules! ensure_finite {
    ($val:expr, $label:expr, $x:expr, $y:expr, $step:expr) => {
        if !$val.is_finite() {
            panic!(
                "Detected non-finite value for {} at cell ({}, {}) on step {}: {}",
                $label, $x, $y, $step, $val
            );
        }
    };
}
use ensure_finite;

// -----------------------------------------------------------------------------
// Neighbour offsets for axial "columns-lined" hex layout
// -----------------------------------------------------------------------------
// Even and odd columns have mirrored vertical diagonals.
// We store them once to avoid repeat allocation in tight loops.

const NEIGH_OFFSETS_EVEN: [(i16, i16); 6] = [
    (1, 0),  // 4 o'clock (east)
    (0, 1),  // 6 o'clock (south)
    (-1, 0), // 8 o'clock (west)
    (0, -1), // 12 o'clock (north)
    (-1, -1),// 10 o'clock (north-west)
    (1, -1), // 2 o'clock (north-east)
];

const NEIGH_OFFSETS_ODD: [(i16, i16); 6] = [
    (1, 0),  // 2 o'clock (east)
    (0, 1),  // 6 o'clock (south)
    (-1, 0), // 10 o'clock (west)
    (0, -1), // 12 o'clock (north)
    (-1, 1), // 8 o'clock (south-west)
    (1, 1),  // 4 o'clock (south-east)
];

struct Hex {
    coordinate: (usize, usize),
    elevation: f32, // Feet
    water_depth: f32, // Feet of water currently stored in this hex
    suspended_load: f32, // Feet of sediment stored in water column
    rainfall: f32, // Feet of rainfall added to this hex per step
    // TODO: an original_land flag would be useful for logging.
}

// This can be done more cleanly with floor division, but the
// docs describe div_floor as part of an experimental API.
fn hex_distance(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    let dx = (x1 - x2).abs();
    if dx % 2 == 0 {
        let diagonal_only_y_min = y1 - dx / 2;
        let diagonal_only_y_max = y1 + dx / 2;
        return dx + 0.max(diagonal_only_y_min - y2).max(y2 - diagonal_only_y_max);
    } else {
        let diagonal_only_y_min = y1 - (dx - x1 % 2) / 2;
        let diagonal_only_y_max = y1 + (dx + x1 % 2) / 2;
        return dx + 0.max(diagonal_only_y_min - y2).max(y2 - diagonal_only_y_max);
    }
}

fn hex_coordinates_to_cartesian(x: i32, y: i32) -> (f32, f32) {
    let dx = x as f32 * HEX_FACTOR;
    let mut dy = y as f32;
    if x % 2 == 1 {
        dy += 0.5;
    }
    (dx, dy)
}

fn cartesian_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

fn hex_distance_pythagorean(x1: i32, y1: i32, x2: i32, y2: i32) -> f32 {
    let (dx, dy) = hex_coordinates_to_cartesian(x1, y1);
    let (dx2, dy2) = hex_coordinates_to_cartesian(x2, y2);
    return cartesian_distance(dx, dy, dx2, dy2);
}

fn elevation_to_color(elevation: f32) -> Rgb<u8> {
    if elevation < SEA_LEVEL {
        let normalized_elevation = ((elevation + ABYSSAL_PLAINS_MAX_DEPTH) / (ABYSSAL_PLAINS_MAX_DEPTH)).min(1.0);
        if normalized_elevation < 0.0 {
            // Black, to mark where errosion has lowered elevation below the lowest possible initial elevation.
            Rgb([0, 0, 0])
        } else {
            // Blue to light blue
            let red = 0;
            let green = (255.0 * normalized_elevation) as u8;
            let blue = 255;
            Rgb([red, green, blue])
        }
    } else {
        // Land: green -> yellow -> orange -> red -> brown -> white
        let land_height = elevation;
        // TODO: I expect the 256.0 / 255.0 is compensating for a math error somewhere else.
        let max_land_height = MAX_ELEVATION * 256.0 / 255.0;
        let normalized_height = (land_height / max_land_height).min(1.0);
        
        if normalized_height < 0.15 {
            // Green to yellow
            let factor = normalized_height / 0.15;
            let red = (255.0 * factor) as u8;
            let green = 255;
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.3 {
            // Yellow (255,255,0) → Orange (255,127,0)
            let factor = (normalized_height - 0.15) / 0.15; // 0..1
            let red   = 255;
            let green = (127.0 + 128.0 * (1.0 - factor)) as u8; // 255→127
            let blue  = 0;
            Rgb([red, green, blue])
        // TODO: The later stages of this transition don't subjectively look right to me,
        // is there some standard way to do this?
        } else if normalized_height < 0.5 {
            // Orange (255,127,0) to Red (255,0,0)
            let factor = (normalized_height - 0.3) / 0.2; // 0..1
            let red = 255;
            let green = (127.0 * (1.0 - factor)) as u8; // 127→0
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.7 {
            // Red to brown
            let factor = (normalized_height - 0.5) / 0.2;
            let red = 62 + (193.0 * (1.0 - factor)) as u8; // 255→62
            let green = 28 * factor as u8;
            let blue = 0;
            Rgb([red, green, blue])
        } else {
            // Brown to white
            let factor = (normalized_height - 0.7) / 0.3;
            let red = 62 + (193.0 * factor) as u8; // 62→255
            let green = 28 + (237.0 * factor) as u8;
            let blue = (255.0 * factor) as u8;
            // TODO: I seem to be getting some weird magenta dots in the output, not sure why.
            Rgb([red, green, blue])
        }
    }
}

fn upload_hex_data(hex_map: &Vec<Vec<Hex>>, gpu_sim: &GpuSimulation) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_PIXELS as usize;
    let mut gpu_data: Vec<HexGpu> = Vec::with_capacity(width * height);
    for row in hex_map.iter() {
        for h in row {
            gpu_data.push(HexGpu {
                elevation: h.elevation,
                water_depth: h.water_depth,
                suspended_load: h.suspended_load,
                rainfall: h.rainfall,
            });
        }
    }
    gpu_sim.upload_data(&gpu_data);
}

fn download_hex_data(gpu_sim: &GpuSimulation, hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;
    let gpu_hex_data = gpu_sim.download_hex_data();
    for (idx, h) in gpu_hex_data.iter().enumerate() {
        let y = idx / width;
        let x = idx % width;
        let cell = &mut hex_map[y][x];
        cell.elevation = h.elevation;
        cell.water_depth = h.water_depth;
        cell.suspended_load = h.suspended_load;
    }
}

fn let_slopes_settle(hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);
    gpu_sim.resize_min_buffers(width, height);

    upload_hex_data(hex_map, &gpu_sim);

    // TODO: Keep an eye on whether 20 iterations is enough.
    for _ in 0..20 {
        gpu_sim.run_repose_step(width, height);
    }

    download_hex_data(&gpu_sim, hex_map);
}

fn fill_sea(hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    for y in 0..height {
        for x in 0..width {
            let cell = &mut hex_map[y][x];
            if cell.elevation < SEA_LEVEL {
                cell.water_depth = SEA_LEVEL - cell.elevation;
            }
        }
    }
}

fn simulate_rainfall(
    hex_map: &mut Vec<Vec<Hex>>,
    steps: u32,
    river_y: usize,
    river_outlet_x: usize,
) -> f32 {
    let water_start = Instant::now();

    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    // ---------------------------------------------------------
    // GPU helper initialisation (only used for rainfall phase)
    // ---------------------------------------------------------
    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);
    gpu_sim.resize_min_buffers(width, height);

    let mut total_outflow = 0.0f32;
    let mut total_sediment_in = 0.0f32;
    let mut total_sediment_out = 0.0f32;

    let mut current_sea_level = SEA_LEVEL;

    upload_hex_data(hex_map, &gpu_sim);

    println!(
        "Calculated constants: NORTH_DESERT_WIDTH {}  NE_BASIN_WIDTH {}  TOTAL_LAND_WIDTH {}  TOTAL_SEA_WIDTH {}  SW_RANGE_WIDTH {}",
        NORTH_DESERT_WIDTH, NE_BASIN_WIDTH, TOTAL_LAND_WIDTH, TOTAL_SEA_WIDTH, SW_RANGE_WIDTH
    );
    println!(
        "  CONTINENTAL_SHELF_INCREMENT {}  CONTINENTAL_SLOPE_INCREMENT {}  ABYSSAL_PLAINS_INCREMENT {}",
        CONTINENTAL_SHELF_INCREMENT, CONTINENTAL_SLOPE_INCREMENT, ABYSSAL_PLAINS_INCREMENT
    );
    println!(
        "  SEA_LEVEL {}  NORTH_DESERT_HEIGHT {}  CENTRAL_HIGHLAND_HEIGHT {}  SOUTH_MOUNTAINS_HEIGHT {}",
        SEA_LEVEL, NORTH_DESERT_HEIGHT, CENTRAL_HIGHLAND_HEIGHT, SOUTH_MOUNTAINS_HEIGHT
    );
    println!(
        "  RAINFALL_FACTOR {}  EVAPORATION_FACTOR {}",
        RAINFALL_FACTOR,
        EVAPORATION_FACTOR
    );
    println!(
        "  RIVER_Y {}  RIVER_SOURCE_X {}  RIVER_OUTLET_X {}",
        RIVER_Y,
        RIVER_SOURCE_X,
        river_outlet_x
    );

    for _step in 0..steps {
        // Mass balance stats per step
        let mut step_outflow = 0.0f32;
        let mut step_sediment_in = 0.0f32;
        let mut step_sediment_out = 0.0f32;

        if _step % (WIDTH_HEXAGONS as u32 * LOG_ROUNDS) == 0 {
            // Download hex data after all GPU passes for CPU-side logic
            let gpu_hex_data = gpu_sim.download_hex_data();
            for (idx, h) in gpu_hex_data.iter().enumerate() {
                let y = idx / width;
                let x = idx % width;
                let cell = &mut hex_map[y][x];
                cell.elevation = h.elevation;
                cell.water_depth = h.water_depth;
                cell.suspended_load = h.suspended_load;
            }

            let rainfall_added: f32 = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.elevation > SEA_LEVEL).fold(0.0, |acc, h| acc + h.rainfall))
                .sum();

            let cells_above_sea_level: usize = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.elevation > SEA_LEVEL).count())
                .sum();

            let (water_on_land, max_depth) = hex_map
                .par_iter()
                .map(|row| {
                    let mut sum = 0.0f32;
                    let mut row_max = 0.0f32;
                    for h in row {
                        if h.elevation > SEA_LEVEL {
                            let d = h.water_depth;
                            sum += d;
                            if d > row_max {
                                row_max = d;
                            }
                        }
                    }
                    (sum, row_max)
                })
                .reduce(
                    || (0.0f32, 0.0f32),
                    |acc, val| {
                        (
                            acc.0 + val.0,
                            acc.1.max(val.1),
                        )
                    },
                );

            // Experimenting with h.elevation > SEA_LEVEL or h.water_depth <= WATER_THRESHOLD.
            let westernmost_land_hex = hex_map.par_iter().map(|row| {
                row.iter().filter(|h| h.water_depth <= WATER_THRESHOLD).min_by_key(|h| h.coordinate.0)
            }).flatten().min_by_key(|h| h.coordinate.0);

            let mean_depth = water_on_land / cells_above_sea_level as f32;

            let wet_cells: usize = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.elevation > SEA_LEVEL && h.water_depth > WATER_THRESHOLD).count())
                .sum();

            let wet_cells_percentage = wet_cells as f32 / cells_above_sea_level as f32 * 100.0;

            let source_hex = &hex_map[RIVER_Y][RIVER_SOURCE_X];
            let outlet_hex = &hex_map[RIVER_Y][river_outlet_x];
            let target_delta_hex = &hex_map[RIVER_Y][river_outlet_x - 231];

            // The more the hex we want to be the edge of the delta is dry, the more we want
            // we want to bring the sea level up to match.
            // TODO: Is there an inexpensive way to download this every step?
            // if target_delta_hex.elevation - target_delta_hex.water_depth > current_sea_level {
            //     current_sea_level = target_delta_hex.elevation - target_delta_hex.water_depth;
            // }

            let round = _step / (WIDTH_HEXAGONS as u32);

            let (min_elevation, max_elevation) = hex_map.par_iter().map(|row| {
                let mut row_min = f32::INFINITY;
                let mut row_max = f32::NEG_INFINITY;
                for h in row {
                    if h.coordinate.0 > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                        continue;
                    }
                    if h.elevation < row_min {
                        row_min = h.elevation;
                    }
                    if h.elevation > row_max {
                        row_max = h.elevation;
                    }
                }
                (row_min, row_max)
            }).reduce(|| (f32::INFINITY, f32::NEG_INFINITY), |acc, val| {
                (acc.0.min(val.0), acc.1.max(val.1))
            });

            println!(
                "Round {:.0}: water in {:.3}  stored {:.0}  mean depth {:.2} ft  max depth {:.2} ft  wet {:} ({:.1}%)  source elevation {:.2} ft  outlet elevation {:.2} ft  target delta elevation {:.2} ft  westernmost land X {}",
                round,
                (rainfall_added + RIVER_WATER_PER_STEP),
                water_on_land,
                mean_depth,
                max_depth,
                wet_cells,
                wet_cells_percentage,
                source_hex.elevation,
                outlet_hex.elevation,
                target_delta_hex.elevation,
                westernmost_land_hex.map_or(0, |h| h.coordinate.0),
            );
            println!("  min elevation: {:.2} ft  max elevation: {:.2} ft  time: {:?}", min_elevation, max_elevation, water_start.elapsed());

            let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];
            render_frame(hex_map, &mut frame_buffer, current_sea_level, true);
            save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

            render_frame(hex_map, &mut frame_buffer, current_sea_level, false);
            save_buffer_png("terrain.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
        }

        gpu_sim.run_rainfall_step(width * height, current_sea_level);

        // TODO: Possible to get water / sediment inflow from GPU for diagnostic purposes?
        // If we don't delete this entirely.
        // let source_idx = (river_y * width + (width - 1)) as u32;
        // gpu_sim.run_river_source_update(source_idx);

        // GPU water routing
        gpu_sim.run_water_routing_step(width, height, FLOW_FACTOR, MAX_FLOW);
        gpu_sim.run_scatter_step(width, height);

        // TODO: Similar to the river source updater, is it possible to get water / sediment outflows
        // from GPU for diagnostic purposes?
        gpu_sim.run_ocean_boundary(width, height, current_sea_level);

        // GPU min-slope + erosion/deposition passes
        gpu_sim.run_min_neigh_step(width, height);
        gpu_sim.run_erosion_step(width, height);

        gpu_sim.run_repose_step(width, height);
    }

    download_hex_data(&gpu_sim, hex_map);

    let water_remaining: f32 = hex_map
        .iter()
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth)
        .sum();

    let water_remaining_north: f32 = hex_map
        .iter()
        .take(NORTH_DESERT_HEIGHT)
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth)
        .sum();

    let water_remaining_ne_basin: f32 = hex_map
        .iter()
        .take(NORTH_DESERT_HEIGHT)
        .flat_map(|row| row.iter().skip(TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH))
        .map(|h| h.water_depth)
        .sum();

    let water_remaining_central: f32 = hex_map
        .iter()
        .skip(NORTH_DESERT_HEIGHT)
        .take(CENTRAL_HIGHLAND_HEIGHT)
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth)
        .sum();

    // Experimenting with h.elevation > current_sea_level or h.water_depth <= WATER_THRESHOLD.
    let westernmost_land_hex = hex_map.par_iter().map(|row| {
        row.iter().filter(|h| h.water_depth <= WATER_THRESHOLD).min_by_key(|h| h.coordinate.0)
    }).flatten().min_by_key(|h| h.coordinate.0);

    let westernmost_land_hex_north = hex_map.par_iter().take(NORTH_DESERT_HEIGHT).map(|row| {
        row.iter().filter(|h| h.water_depth <= WATER_THRESHOLD).min_by_key(|h| h.coordinate.0)
    }).flatten().min_by_key(|h| h.coordinate.0);

    let westernmost_land_hex_central = hex_map.par_iter().skip(NORTH_DESERT_HEIGHT).take(CENTRAL_HIGHLAND_HEIGHT).map(|row| {
        row.iter().filter(|h| h.water_depth <= WATER_THRESHOLD).min_by_key(|h| h.coordinate.0)
    }).flatten().min_by_key(|h| h.coordinate.0);

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft-hexes,",
        steps,
        total_outflow
    );

    println!(" westernmost land: {}, westernmost land north: {}, westernmost land south: {},",
        westernmost_land_hex.map_or(0, |h| h.coordinate.0),
        westernmost_land_hex_north.map_or(0, |h| h.coordinate.0),
        westernmost_land_hex_central.map_or(0, |h| h.coordinate.0),
    );

    println!(
        " water remaining on land: {:.2} ft-hexes, water remaining north: {:.2} ft-hexes, water remaining NE basin: {:.2} ft-hexes,  water remaining central: {:.2} ft-hexes, sediment in {:.1},  sediment out {:.1}",
        water_remaining,
        water_remaining_north,
        water_remaining_ne_basin,
        water_remaining_central,
        total_sediment_in,
        total_sediment_out
    );

    current_sea_level
}

fn save_buffer_png(path: &str, buffer: &[u32], width: u32, height: u32) {
    let mut img: RgbImage = RgbImage::new(width, height);
    for (idx, pixel) in buffer.iter().enumerate() {
        let r = ((pixel >> 16) & 0xFF) as u8;
        let g = ((pixel >> 8) & 0xFF) as u8;
        let b = (pixel & 0xFF) as u8;
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        img.put_pixel(x, y, Rgb([r, g, b]));
    }
    img.save(path).expect("Failed to save image");
}

fn save_png(path: &str, hex_map: &Vec<Vec<Hex>>, sea_level: f32) {
    // Create the visualization image
    let mut img = ImageBuffer::new(WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
    
    // For each pixel, find the nearest hex and use its elevation
    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            // Convert pixel coordinates to hex coordinates
            // This is a simple mapping - you might want to adjust this based on your hex layout
            let hex_x = (x as f32 / HEX_FACTOR) as usize;
            let hex_y = y;
            
            // Clamp to valid hex coordinates
            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);
            
            // Get the elevation from the hex map
            let elevation = hex_map[hex_y as usize][hex_x as usize].elevation;
            
            // Convert elevation to color
            let color = elevation_to_color(elevation - sea_level);
            
            // Set the pixel
            img.put_pixel(x as u32, y as u32, color);
        }
    }

    img.save(path).expect("Failed to save image");
}

// Renders current hex_map state into an RGB buffer (u32 per pixel)
fn render_frame(hex_map: &Vec<Vec<Hex>>, buffer: &mut [u32], sea_level: f32, show_water: bool) {
    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            let hex_x = ((x as f32) / HEX_FACTOR) as usize;
            let hex_y = y;

            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);

            let hex = &hex_map[hex_y as usize][hex_x as usize];
            // Choose colour – highlight water depth strongly so it stands out
            let color = if show_water && hex.water_depth > WATER_THRESHOLD {
                // Strong blue for water for debugging
                let blue = 255u8;
                let g = 0u8;
                let r = 0u8;
                (r as u32) << 16 | (g as u32) << 8 | (blue as u32)
            } else {
                let Rgb([r, g, b]) = elevation_to_color(hex.elevation - sea_level);
                let r = (r as f32 * 0.4) as u8;
                let g = (g as f32 * 0.4) as u8;
                let b = (b as f32 * 0.4) as u8;
                (r as u32) << 16 | (g as u32) << 8 | (b as u32)
            };
            buffer[(y as usize) * (WIDTH_PIXELS as usize) + (x as usize)] = color;
        }
    }
}

fn get_erruption_elevation(target_elevation: f32) -> f32 {
    let mut erruption_elevation = target_elevation;
    let mut current_target_elevation = target_elevation - HEX_SIZE;
    let mut ring = 1.0;
    while current_target_elevation > 0.0 {
        erruption_elevation += ring * 6.0 * current_target_elevation;
        current_target_elevation -= HEX_SIZE;
        ring += 1.0;
    }
    erruption_elevation
}

fn get_perlin_noise(perlin: &Perlin, input: f64, period: f64) -> f32 {
    (perlin.get([input / period]) as f32 + 1.0) / 2.0
}

fn get_perlin_noise_for_hex(perlin: &Perlin, x: f64, y: f64, period: f64) -> f32 {
    (perlin.get([x * HEX_FACTOR as f64 / period, y / period]) as f32 + 1.0) / 2.0
}

fn get_sea_deviation(perlin: &Perlin, y: f64, period: f64) -> usize {
    ((get_perlin_noise(perlin, y, period) - 0.5) * COAST_WIDTH as f32) as usize
}

fn main() {
    // Allow user to override number of rounds via command-line: first positional arg is rounds, e.g. `cargo run --release -- 2000`
    let rounds: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_ROUNDS);

    let mut hex_map = Vec::new();
    let mut rng = rand::thread_rng();
    let perlin = Perlin::new(rng.gen_range(0..u32::MAX));
    // Two degrees latitude, in half-mile hexes.
    let transition_period = ONE_DEGREE_LATITUDE_MILES as f64 * 4.0;
    let sea_deviation_for_river_y = get_sea_deviation(&perlin, RIVER_Y as f64, HEIGHT_PIXELS as f64 / 1.5);
    let river_outlet_x = TOTAL_SEA_WIDTH - sea_deviation_for_river_y;

    // Time hex map creation
    let hex_start = Instant::now();

    for y in 0..HEIGHT_PIXELS {
        hex_map.push(Vec::new());
        let distance_from_river_y = (y as i16 - RIVER_Y as i16).abs();
        let sea_deviation = get_sea_deviation(&perlin, y as f64, HEIGHT_PIXELS as f64 / 1.5);

        for x in 0..WIDTH_HEXAGONS {
            let mut elevation = 0.0;
            let mut distance_from_coast = 0;
            let adjusted_y = y as f64 + (x % 2) as f64 * 0.5;

            if x + sea_deviation < ABYSSAL_PLAINS_WIDTH {
                elevation = -1.0 * ABYSSAL_PLAINS_MIN_DEPTH - (ABYSSAL_PLAINS_WIDTH - (x + sea_deviation)) as f32 * ABYSSAL_PLAINS_INCREMENT;
            } else if x + sea_deviation >= ABYSSAL_PLAINS_WIDTH && x + sea_deviation < ABYSSAL_PLAINS_WIDTH + CONTINENTAL_SLOPE_WIDTH {
                elevation = (x + sea_deviation - ABYSSAL_PLAINS_WIDTH) as f32 * CONTINENTAL_SLOPE_INCREMENT - ABYSSAL_PLAINS_MIN_DEPTH;
            } else if x + sea_deviation < TOTAL_SEA_WIDTH {
                elevation = (x + sea_deviation - ABYSSAL_PLAINS_WIDTH - CONTINENTAL_SLOPE_WIDTH) as f32 * CONTINENTAL_SHELF_INCREMENT - CONTINENTAL_SHELF_DEPTH;
            } else {
                distance_from_coast = x + sea_deviation - TOTAL_SEA_WIDTH;
                let map_third_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, HEIGHT_PIXELS as f64 / 3.0);
                let transition_period_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, transition_period);
                let coastal_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, COAST_WIDTH as f64);
                let perlin_noise = ((transition_period_noise + coastal_noise + map_third_noise) / 3.0).powf(3.0_f32.log2());

                if y < NORTH_DESERT_HEIGHT - (transition_period * 2.0) as usize {
                    // This is to prevent the river outlet from being too far north.
                    // TODO: Maybe fade this out once we get very far north?
                    // let coast_factor = (1.0 - distance_from_coast as f32 * 2.0 / COAST_WIDTH as f32).max(0.0);
                    let factor = ((NORTH_DESERT_HEIGHT - (transition_period * 2.0) as usize - y) as f32 / (transition_period as f32)).min(1.0);
                    elevation = perlin_noise * NORTH_DESERT_MAX_ELEVATION + (1.0 - perlin_noise) * NORTH_DESERT_MAX_ELEVATION * factor * 0.25;
                } else if y < NORTH_DESERT_HEIGHT {
                    let (cx1, cy1) = hex_coordinates_to_cartesian(x as i32, y as i32);
                    let (cx2, cy2) = hex_coordinates_to_cartesian(TOTAL_SEA_WIDTH as i32 - sea_deviation_for_river_y as i32, RIVER_Y as i32);
                    // Area is oval-shaped, not circular, with the longer axis running east-west.
                    let factor = (cartesian_distance(0.0, cy1, (cx2 - cx1) / 2.0, cy2) / (transition_period as f32 / 2.0)).min(1.0);
                    elevation = perlin_noise * NORTH_DESERT_MAX_ELEVATION * factor;
                } else if y < NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                    // TODO: Similar to making sure the river isn't too far north, we should take steps to make sure the delta isn't too far south.
                    let coast_y = COAST_WIDTH as f32 * HEX_SIZE;
                    let coast_factor = (distance_from_coast as f32 * 2.0 / COAST_WIDTH as f32).max((coast_y - (NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT - y) as f32) as f32 * 2.0 / (COAST_WIDTH as f32 * HEX_SIZE));
                    // Faster transition because it's a less dramatic change.
                    let factor = ((y - NORTH_DESERT_HEIGHT) as f32 / transition_period as f32 * 2.0).min(coast_factor).min(1.0);
                    let max_elevation = ((CENTRAL_HIGHLAND_MAX_ELEVATION - NORTH_DESERT_MAX_ELEVATION) * factor + NORTH_DESERT_MAX_ELEVATION);
                    elevation = perlin_noise * max_elevation;

                    if y < NORTH_DESERT_HEIGHT + (transition_period * 2.0) as usize {
                        let secondary_factor = ((NORTH_DESERT_HEIGHT as f32 + transition_period as f32 - y as f32) / (transition_period as f32)).abs().clamp(0.0, 1.0);
                        elevation += (1.0 - perlin_noise) * (max_elevation / 6.0) * (1.0 - secondary_factor);
                    }
                } else {
                    // Should probably be renamed "south mountains"
                    let coast_factor = distance_from_coast as f32 * 2.0 / COAST_WIDTH as f32;
                    let factor = ((y - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT) as f32 / transition_period as f32).min(coast_factor).min(1.0);
                    elevation = perlin_noise * ((SE_MOUNTAINS_MAX_ELEVATION - CENTRAL_HIGHLAND_MAX_ELEVATION) * factor + CENTRAL_HIGHLAND_MAX_ELEVATION);
                }
            }

            // let mut bumper_factor = 0.0;
            // if y > NORTH_DESERT_HEIGHT - BUMPER_RANGE && y < NORTH_DESERT_HEIGHT + BUMPER_RANGE {
            //     bumper_factor = 1.0 - (y as i32 - NORTH_DESERT_HEIGHT as i32).abs() as f32 / (BUMPER_RANGE as f32);
            // } else if y > NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT - BUMPER_RANGE && y < NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT + BUMPER_RANGE {
            //     bumper_factor = 1.0 - (y as i32 - NORTH_DESERT_HEIGHT as i32 - CENTRAL_HIGHLAND_HEIGHT as i32).abs() as f32 / (BUMPER_RANGE as f32);
            // }
            // elevation += BUMPER_MAX_ELEVATION * bumper_factor * (distance_from_coast as f32 / (TOTAL_LAND_WIDTH as f32));

            // TODO: More realistic seafloor depth, and fix islands.
            // Special features:
            if x == BIG_VOLCANO_X && y == RIVER_Y {
                // A "volcano" sitting in the river's path. Initially a very tall one-hex column but the angle of repose logic will collapse it,
                // usually into a cone.
                // TODO: Make this an event that happens as rainfall is happening, after enough time that I can pick the spot in the desired area with the most water.
                elevation += get_erruption_elevation(HEX_SIZE * 5.0);   
            } else if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE && y <= NE_BASIN_HEIGHT + NE_BASIN_FRINGE {
                // The northeast basin.
                // TODO: Refactor this so I'm not doing the goofy thing in the rainfall step.
                elevation = NORTH_DESERT_MAX_ELEVATION + RANDOM_ELEVATION_FACTOR;
                if distance_from_river_y > 0 {
                    let factor = (distance_from_river_y as f32 / (NORTH_DESERT_HEIGHT as f32 - RIVER_Y as f32)).min(1.0);
                    elevation += RANDOM_ELEVATION_FACTOR * factor;
                }
            } else if x == ISLAND_CHAIN_X {
                // Need to figure out how to do this without causing out-of-control erosion of the sea floor.
                // if y == FIRST_ISLAND_Y {
                //     elevation += get_erruption_elevation(SEA_LEVEL - elevation + FIRST_ISLAND_MAX_ELEVATION);
                // } else if y == SECOND_ISLAND_Y {
                //     elevation += get_erruption_elevation(SEA_LEVEL - elevation + SECOND_ISLAND_MAX_ELEVATION);
                // }
            } else if hex_distance(x as i32, y as i32, RING_VALLEY_X as i32, RING_VALLEY_Y as i32) == RING_VALLEY_RADIUS as i32 {
                // TODO: Use Pythagorean distance, 8 hex radius, also hill climb to find a local maxima (probably requires moving to post-generation step).
                elevation += RING_VALLEY_ELEVATION_BONUS;
            }
            // TODO: seaside cliff just north of 34 degrees latitude. Make it 512 feet at the highest point, like the Athenian acropolis.
            if x + sea_deviation <= TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE as usize {
                elevation += (get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, 6.0) - 0.5) * RANDOM_ELEVATION_FACTOR * 3.0;
            }

            let mut rain_class = 0;
            if x + sea_deviation > TOTAL_SEA_WIDTH && distance_from_coast < COAST_WIDTH {
                rain_class += 1;
            }
            if y > NORTH_DESERT_HEIGHT {
                rain_class += 1;
            }
            if y > NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                rain_class += 1;
            }
            if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                if y <= NE_BASIN_HEIGHT {
                    rain_class = 4;
                    if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH + NE_BASIN_FRINGE {
                        elevation = NORTH_DESERT_MAX_ELEVATION + RANDOM_ELEVATION_FACTOR;
                    }
                } else {
                    // Set lower so we can see the actual maximum elevation generated by the Perlin noise.
                    elevation = MAX_ELEVATION;
                    rain_class = -1;
                }
            }

            // Case switch statement to set rainfall based on rain_class:
            let rainfall = match rain_class {
                0 => VERY_LOW_RAIN,
                1 => LOW_RAIN,
                2 => MEDIUM_RAIN,
                3 => HIGH_RAIN,
                4 => VERY_HIGH_RAIN,
                // default error case
                _ => 0.0,
            };

            if distance_from_river_y == 0 {
                elevation = elevation.min(NORTH_DESERT_MAX_ELEVATION + RANDOM_ELEVATION_FACTOR);
            } else {
                // Not sure why I keep needing these little adjustments to avoid magenta
                elevation = elevation.min(MAX_ELEVATION * 255.0 / 256.0);
            }
            

            hex_map[y as usize].push(Hex {
                coordinate: (x, y),
                elevation,
                water_depth: 0.0,
                suspended_load: 0.0,
                rainfall: rainfall * RAINFALL_FACTOR,
            });
        }
    }

    let_slopes_settle(&mut hex_map);
    fill_sea(&mut hex_map);
    // prefill_basins(&mut hex_map);

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    let total_steps = (WIDTH_HEXAGONS as u32) * rounds;
    let final_sea_level = simulate_rainfall(&mut hex_map, total_steps, RIVER_Y, river_outlet_x);

    // TODO: This isn't working, should fix.
    // Count final blue pixels for quick sanity check
    let final_blue = frame_buffer
        .iter()
        .filter(|&&px| (px & 0x0000FF) == 0x0000FF && (px >> 16 & 0xFF) == 0 && (px >> 8 & 0xFF) == 0)
        .count();
    println!("Final blue pixels: {}", final_blue);

    let hex_duration = hex_start.elapsed();
    println!("Hex map creation took: {:?}", hex_duration);

    // Time PNG conversion
    let png_start = Instant::now();
    render_frame(&mut hex_map, &mut frame_buffer, final_sea_level, true);
    save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

    render_frame(&mut hex_map, &mut frame_buffer, final_sea_level, false);
    save_buffer_png("terrain.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

    let save_duration = png_start.elapsed();
    println!("Image rendering and saving took: {:?}", save_duration);
    println!("Terrain visualization saved as terrain.png");
}