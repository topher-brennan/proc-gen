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
use clap::Parser;

#[derive(Parser)]
#[command(name = "proc-gen")]
#[command(about = "A procedural terrain generation simulation")]
struct Args {
    /// Number of simulation rounds to run
    #[arg(long, default_value_t = DEFAULT_ROUNDS)]
    rounds: u32,
    
    /// Random seed for terrain generation
    #[arg(long)]
    seed: Option<u32>,
}

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

// Prefill basins with water
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

// Neighbor offsets for "columns-lined" hex layout
const NEIGH_OFFSETS_EVEN: [(i16, i16); 6] = [
    (1, 0),  // 4 o'clock
    (0, 1),  // 6 o'clock
    (-1, 0), // 8 o'clock
    (0, -1), // 12 o'clock
    (-1, -1),// 10 o'clock
    (1, -1), // 2 o'clock
];

const NEIGH_OFFSETS_ODD: [(i16, i16); 6] = [
    (1, 0),  // 2 o'clock
    (0, 1),  // 6 o'clock
    (-1, 0), // 10 o'clock
    (0, -1), // 12 o'clock
    (-1, 1), // 8 o'clock
    (1, 1),  // 4 o'clock
];

struct Hex {
    coordinate: (usize, usize),
    elevation: f32, // Feet
    water_depth: f32, // Feet of water currently stored in this hex
    suspended_load: f32, // Feet of sediment stored in water column
    rainfall: f32, // Feet of rainfall added to this hex per step
    erosion_multiplier: f32,
    original_land: bool,
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
            let factor = ((normalized_height - 0.7) / 0.3).clamp(0.0, 1.0);
            let red = 62 + (193.0 * factor) as u8; // 62→255
            let green = 28 + ((237.0 * factor) as u8).clamp(0, 255 - 28);
            let blue = (255.0 * factor) as u8;
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
                residual_elevation: 0.0,
                erosion_multiplier: h.erosion_multiplier,
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
        cell.elevation = h.elevation + h.residual_elevation;
        cell.water_depth = h.water_depth;
        cell.suspended_load = h.suspended_load;
        cell.erosion_multiplier = h.erosion_multiplier;
    }
}

fn let_slopes_settle(hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);
    gpu_sim.resize_min_buffers(width, height);

    upload_hex_data(hex_map, &gpu_sim);

    for _ in 0..10 {
        // This is the only place this function is called,
        // may create opportunities for refactoring.
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

fn simulate_erosion(
    hex_map: &mut Vec<Vec<Hex>>,
    steps: u32,
    river_outlet_x: usize,
) -> f32 {
    let water_start = Instant::now();

    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);
    gpu_sim.resize_min_buffers(width, height);

    // TODO: Get these working again.
    let mut total_outflow = 0.0f32;
    let mut total_sediment_in = 0.0f32;
    let mut total_sediment_out = 0.0f32;

    // TODO: Vary sea level over time.
    let mut current_sea_level = SEA_LEVEL;

    upload_hex_data(hex_map, &gpu_sim);

    println!(
        "Calculated constants: NORTH_DESERT_WIDTH {}  NE_BASIN_WIDTH {}  TOTAL_LAND_WIDTH {}  TOTAL_SEA_WIDTH {}",
        NORTH_DESERT_WIDTH, NE_BASIN_WIDTH, TOTAL_LAND_WIDTH, TOTAL_SEA_WIDTH
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

    for step in 0..steps {
        // Map's dimensions define what constitutes a "round", since it determines how long it takes
        // for changes on one side of the map to propagate to the other.
        if step % (WIDTH_HEXAGONS.max(HEIGHT_PIXELS) as u32 * LOG_ROUNDS / 1000) == 0 {
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
                    let mut sum = 0.0f64;
                    let mut row_max = 0.0f64;
                    for h in row {
                        if h.elevation > SEA_LEVEL {
                            let d = h.water_depth as f64;
                            sum += d;
                            if d > row_max {
                                row_max = d;
                            }
                        }
                    }
                    (sum, row_max)
                })
                .reduce(
                    || (0.0f64, 0.0f64),
                    |acc, val| {
                        (
                            acc.0 + val.0,
                            acc.1.max(val.1),
                        )
                    },
                );

            let mean_depth = water_on_land / cells_above_sea_level as f64;

            let wet_cells: usize = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.elevation > SEA_LEVEL && h.water_depth > WATER_THRESHOLD).count())
                .sum();

            let wet_cells_percentage = wet_cells as f64 / cells_above_sea_level as f64 * 100.0;

            let source_hex = &hex_map[RIVER_Y][RIVER_SOURCE_X];
            let outlet_hex = &hex_map[RIVER_Y][river_outlet_x];
            let target_delta_hex = &hex_map[RIVER_Y][river_outlet_x - 231];

            // The more the hex we want to be the edge of the delta is dry, the more we want
            // we want to bring the sea level up to match.
            // TODO: Is there an inexpensive way to download this every step?
            // if target_delta_hex.elevation - target_delta_hex.water_depth > current_sea_level {
            //     current_sea_level = target_delta_hex.elevation - target_delta_hex.water_depth;
            // }

            let round = step / (WIDTH_HEXAGONS as u32);

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

            let total_land: f64 = hex_map
                .iter()
                .flat_map(|row| row.iter().filter(|h| h.original_land))
                .map(|h| h.elevation as f64)
                .sum();

            let net_land: f64 = hex_map
                .iter()
                .flat_map(|row| row.iter())
                .map(|h| h.elevation as f64)
                .sum();

            let total_sediment: f64 = hex_map
                .iter()
                .flat_map(|row| row.iter())
                .map(|h| h.suspended_load as f64)
                .sum();

            println!(
                "Round {:.0}: water in {:.3}  stored {:.3}  mean depth {:.3} ft  max depth {:.3} ft  wet {:} ({:.1}%)",
                round,
                rainfall_added,
                water_on_land,
                mean_depth,
                max_depth,
                wet_cells,
                wet_cells_percentage
            );
            println!(
                "  source elevation {:.3} ft  source water depth {:.3} ft  outlet elevation {:.3} ft  target delta elevation {:.3} ft",
                source_hex.elevation,
                source_hex.water_depth,
                outlet_hex.elevation,
                target_delta_hex.elevation,
            );
            println!("  total land (original): {:.3} ft  net land: {:.3} ft  total sediment: {:.3} ft", total_land, net_land, total_sediment);
            println!("  min elevation: {:.3} ft  max elevation: {:.3} ft  time: {:?}", min_elevation, max_elevation, water_start.elapsed());

            let outflows = gpu_sim.download_ocean_outflows(height);
            let total_water_out: f64 = outflows.iter().map(|o| o.water_out as f64).sum();
            let total_sed_out: f64 = outflows.iter().map(|o| o.sediment_out as f64).sum();
            let eros_log = gpu_sim.download_erosion_log(width, height);
            let total_eroded: f64 = eros_log.iter().map(|e| e[0] as f64).sum();
            let total_deposited: f64 = eros_log.iter().map(|e| e[1] as f64).sum();
            println!(
                "  diagnostics: water_out {:.3}  sed_out {:.3}  eroded {:.3}  deposited {:.3}",
                total_water_out, total_sed_out, total_eroded, total_deposited
            );

            let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];
            render_frame(hex_map, &mut frame_buffer, current_sea_level, true);
            save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

            render_frame(hex_map, &mut frame_buffer, current_sea_level, false);
            if step == 0 {
                // For checking for weird behavior as erosion progresses.
                save_buffer_png("terrain_initial.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
            } else {
                save_buffer_png("terrain.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
            }
        } else {
            // TODO: Get the heartbeat to work.
            gpu_sim.heartbeat();
        }

        gpu_sim.run_simulation_step_batched(width, height, current_sea_level, FLOW_FACTOR, MAX_FLOW);
    }

    download_hex_data(&gpu_sim, hex_map);

    let water_remaining: f64 = hex_map
        .iter()
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth as f64)
        .sum();

    let land_remaining: f64 = hex_map
        .iter()
        .flat_map(|row| row.iter().filter(|h| h.original_land))
        .map(|h| h.elevation as f64)
        .sum();

    let water_remaining_north: f64 = hex_map
        .iter()
        .take(NORTH_DESERT_HEIGHT)
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth as f64)
        .sum();

    let water_remaining_ne_basin: f64 = hex_map
        .iter()
        .take(NORTH_DESERT_HEIGHT)
        .flat_map(|row| row.iter().skip(TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH))
        .map(|h| h.water_depth as f64)
        .sum();

    let water_remaining_central: f64 = hex_map
        .iter()
        .skip(NORTH_DESERT_HEIGHT)
        .take(CENTRAL_HIGHLAND_HEIGHT)
        .flat_map(|row| row.iter().filter(|h| h.elevation > current_sea_level))
        .map(|h| h.water_depth as f64)
        .sum();

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
        " water remaining on land: {:.2} ft-hexes, land remaining above sea level: {:.2} ft-hexes, water remaining north: {:.2} ft-hexes, water remaining NE basin: {:.2} ft-hexes,  water remaining central: {:.2} ft-hexes, sediment in {:.1},  sediment out {:.1}",
        water_remaining,
        land_remaining,
        water_remaining_north,
        water_remaining_ne_basin,
        water_remaining_central,
        total_sediment_in,
        total_sediment_out
    );

    current_sea_level
}

fn print_elevation_and_sediment(gpu_sim: &GpuSimulation, step_label: &str) {
    let hex_data = gpu_sim.download_hex_data();
    let sum_elev: f64 = hex_data.iter().map(|h| h.elevation as f64).sum();
    let sum_sed: f64 = hex_data.iter().map(|h| h.suspended_load as f64).sum();
    println!("{}: sum_elev {:.6}  sum_sed {:.6}  sum_mass {:.6}", step_label, sum_elev, sum_sed, sum_elev + sum_sed);
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
            let color = if show_water && hex.water_depth > WATER_THRESHOLD {
                // TODO: Varying shades of blue?
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

// Generates a value between 0 and 1.
fn get_perlin_noise(perlin: &Perlin, input: f64, period: f64) -> f32 {
    (perlin.get([input / period]) as f32 + 1.0) / 2.0
}

// Generates a value between 0 and 1.
fn get_perlin_noise_for_hex(perlin: &Perlin, x: f64, y: f64, period: f64) -> f32 {
    (perlin.get([x * HEX_FACTOR as f64 / period, y / period]) as f32 + 1.0) / 2.0
}

// Generates a value between -COAST_WIDTH/2 and COAST_WIDTH/2.
fn get_sea_deviation(perlin: &Perlin, y: f64, period: f64) -> usize {
    ((get_perlin_noise(perlin, y, period) - 0.5) * COAST_WIDTH as f32) as usize
}

// Generates a value between -12 and 12.
fn get_land_deviation(perlin: &Perlin, x: f64, y: f64, period: f64) -> i16 {
    ((get_perlin_noise_for_hex(perlin, x, y, period) - 0.5) * 24.0 * 2.0) as i16
}

fn get_rainfall(rain_class: i32) -> f32 {
    match rain_class {
        0 => VERY_LOW_RAIN,
        1 => LOW_RAIN,
        2 => MEDIUM_RAIN,
        3 => HIGH_RAIN,
        4 => VERY_HIGH_RAIN,
        _ => 0.0,
    }
}

fn main() {
    let args = Args::parse();
    
    let rounds = args.rounds;
    let seed = args.seed.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..u32::MAX)
    });

    let elevation_adjustment = rounds as f32 * 10.0 * RAINFALL_FACTOR;
    let adj_south_mountains_max_elevation = SOUTH_MOUNTAINS_MAX_ELEVATION * (1.0 + elevation_adjustment);
    let adj_central_highland_max_elevation = CENTRAL_HIGHLAND_MAX_ELEVATION * (1.0 + elevation_adjustment * LOW_RAIN / MEDIUM_RAIN);
    let adj_north_desert_max_elevation = NORTH_DESERT_MAX_ELEVATION * (1.0 + elevation_adjustment * VERY_LOW_RAIN / MEDIUM_RAIN);

    let mut hex_map = Vec::new();
    let mut rng = rand::thread_rng();

    println!("Seed: {}", seed);

    let perlin = Perlin::new(seed);
    let transition_period = ONE_DEGREE_LATITUDE_MILES as f64 * 2.0;
    let sea_deviation_for_river_y = get_sea_deviation(&perlin, RIVER_Y as f64, HEIGHT_PIXELS as f64 / 1.5);
    
    let river_outlet_x = TOTAL_SEA_WIDTH - sea_deviation_for_river_y;
    let land_deviation_for_outlet = get_land_deviation(&perlin, river_outlet_x as f64, RIVER_Y as f64, 96.0);

    // Time hex map creation
    let hex_start = Instant::now();

    for y in 0..HEIGHT_PIXELS {
        hex_map.push(Vec::new());
        let distance_from_river_y = (y as i16 - RIVER_Y as i16).abs();
        let sea_deviation = get_sea_deviation(&perlin, y as f64, HEIGHT_PIXELS as f64 / 1.5);
        let shelf_deviation = get_sea_deviation(&perlin, (y + HEIGHT_PIXELS * 3 / 2) as f64, HEIGHT_PIXELS as f64);
        let cut_factor = -0.06 - get_perlin_noise(&perlin, (y + HEIGHT_PIXELS * 5 / 2) as f64, HEIGHT_PIXELS as f64 / 1.5) * 0.06;

        for x in 0..WIDTH_HEXAGONS {
            let mut elevation = 0.0;
            let mut distance_from_coast = (x + sea_deviation) as f32 - TOTAL_SEA_WIDTH as f32;
            let adjusted_y = y as f64 + (x % 2) as f64 * 0.5;
            let y_deviation = land_deviation_for_outlet - get_land_deviation(&perlin, x as f64, y as f64, 96.0);
            let deviated_y: usize = (y as i16 + y_deviation).max(0) as usize;


            if x + shelf_deviation < ABYSSAL_PLAINS_WIDTH {
                elevation = -1.0 * ABYSSAL_PLAINS_MIN_DEPTH - (ABYSSAL_PLAINS_WIDTH - (x + shelf_deviation)) as f32 * ABYSSAL_PLAINS_INCREMENT;
            } else if x + shelf_deviation >= ABYSSAL_PLAINS_WIDTH && x + shelf_deviation < ABYSSAL_PLAINS_WIDTH + CONTINENTAL_SLOPE_WIDTH {
                elevation = (x + shelf_deviation - ABYSSAL_PLAINS_WIDTH) as f32 * CONTINENTAL_SLOPE_INCREMENT - ABYSSAL_PLAINS_MIN_DEPTH;
            } else if x + sea_deviation < TOTAL_SEA_WIDTH {
                elevation = (x + shelf_deviation - ABYSSAL_PLAINS_WIDTH - CONTINENTAL_SLOPE_WIDTH) as f32 * CONTINENTAL_SHELF_INCREMENT - CONTINENTAL_SHELF_DEPTH;
                elevation = elevation.min(-1.0 * f32::EPSILON);
            } else {
                let map_third_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, HEIGHT_PIXELS as f64 / 3.0);
                let transition_period_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, transition_period);
                let coastal_noise = get_perlin_noise_for_hex(&perlin, x as f64, adjusted_y, COAST_WIDTH as f64);
                let perlin_noise = ((transition_period_noise + coastal_noise + map_third_noise) / 3.0).powf(3.0_f32.log2());

                if deviated_y < NORTH_DESERT_HEIGHT {
                    let (cx1, cy1) = hex_coordinates_to_cartesian(x as i32, deviated_y as i32);
                    let (cx2, cy2) = hex_coordinates_to_cartesian(TOTAL_SEA_WIDTH as i32 - sea_deviation_for_river_y as i32, RIVER_Y as i32);
                    // Area is oval-shaped, not circular, with the longer axis running east-west.
                    // Experiment with ratio of major axis to minor axis, too long might look weird but too short can result in the river
                    // jumping the tracks, so to speak.
                    let factor = (cartesian_distance(0.0, cy1, (cx2 - cx1) / 2.0, cy2) / (transition_period as f32)).min(1.0);
                    elevation = perlin_noise * NORTH_DESERT_MAX_ELEVATION * factor;
                } else if deviated_y < NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                    // Faster transition because it's a less dramatic change.
                    let factor = ((deviated_y - NORTH_DESERT_HEIGHT) as f32 / transition_period as f32 * 2.0).min(1.0);
                    elevation = perlin_noise * ((adj_central_highland_max_elevation - adj_north_desert_max_elevation) * factor + adj_north_desert_max_elevation);
                } else {
                    let factor = ((deviated_y - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT) as f32 / transition_period as f32).min(1.0);
                    elevation = perlin_noise * ((adj_south_mountains_max_elevation - adj_central_highland_max_elevation) * factor + adj_central_highland_max_elevation);
                }

                if deviated_y < RIVER_Y - (transition_period) as usize {
                    // This is to prevent the river outlet from being too far north.
                    let factor = ((RIVER_Y as f32 - transition_period as f32 * 2.0 - deviated_y as f32) / (transition_period as f32)).abs().clamp(0.0, 1.0);
                    elevation = perlin_noise * adj_north_desert_max_elevation + (1.0 - perlin_noise) * adj_north_desert_max_elevation * (1.0 - factor) * 0.3;
                } else if deviated_y < NORTH_DESERT_HEIGHT + transition_period as usize {
                    // Similarly this makes sure the river isn't too far south.
                    let factor = ((NORTH_DESERT_HEIGHT as f32 - deviated_y as f32) / (transition_period as f32)).abs().clamp(0.0, 1.0);
                    elevation += (1.0 - perlin_noise) * (adj_central_highland_max_elevation * 0.3) * (1.0 - factor);
                }
            }

            // This produces a cut whose downward slope is 2 * MOUNTAINS_MAX_ELEVATION / COAST_WIDTH
            // The cut's deepest point is COAST_WIDTH * cut_factor from the coast, at a depth of 2 * MOUNTAINS_MAX_ELEVATION * cut_factor.
            // Then it slopes upward at the multiplicative inverse of the initial slope (if I did my math right).
            if distance_from_coast > COAST_WIDTH as f32 * cut_factor {
                elevation = elevation.min(distance_from_coast * 2.0 / COAST_WIDTH as f32 * adj_south_mountains_max_elevation);
            } else {
                elevation = elevation.min((COAST_WIDTH as f32 * cut_factor - distance_from_coast) * adj_south_mountains_max_elevation / COAST_WIDTH as f32 / 2.0 + adj_south_mountains_max_elevation * cut_factor * 2.0);
            }

            // TODO: More realistic seafloor depth, and fix islands.
            // Special features:
            if x == BIG_VOLCANO_X && y == RIVER_Y {
                // A "volcano" sitting in the river's path. Initially a very tall one-hex column but the angle of repose logic will collapse it,
                // usually into a cone.
                // TODO: Make this an event that happens as rainfall is happening, after enough time that I can pick an interesting spot, e.g.
                // daming the main river, or preventing flow from central area into the north.
                // Also want more realistic slope, 30-35 degrees.
                // elevation += get_erruption_elevation(HEX_SIZE * 5.0);   
            } else if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE && y <= NE_BASIN_HEIGHT + NE_BASIN_FRINGE {
                // The northeast basin.
                // TODO: Refactor this so I'm not doing the goofy thing in the rainfall step.
                elevation = adj_north_desert_max_elevation * 1.01;
                // This specifically doesn't include y-deviation so the river source is exactly where we want it to be.
                if distance_from_river_y != 0 {
                    let factor = ((distance_from_river_y) as f32 / (NORTH_DESERT_HEIGHT as f32 - RIVER_Y as f32)).min(1.0);
                    elevation += (adj_north_desert_max_elevation * 0.01) * factor;
                }
            }

            if x + sea_deviation <= TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE as usize {
                elevation += rng.gen_range(0.0..0.01) * elevation.max(HEX_SIZE as f32);
            }

            let mut rainfall = 0.0;

            if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                if y <= NE_BASIN_HEIGHT {
                    rainfall = VERY_HIGH_RAIN;
                    if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH + NE_BASIN_FRINGE {
                        elevation = adj_north_desert_max_elevation * 1.01;
                    }
                } else {
                    elevation = MAX_ELEVATION;
                }
            } else {
                let mut rain_class = 0;
                if deviated_y > NORTH_DESERT_HEIGHT {
                    rain_class += 1;
                }
                if deviated_y > NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                    rain_class += 1;
                }

                if (distance_from_coast as usize) < COAST_WIDTH - COAST_FRINGE {
                    rainfall = get_rainfall(rain_class + 1);
                } else if (distance_from_coast as usize) > COAST_WIDTH + COAST_FRINGE {
                    rainfall = get_rainfall(rain_class);
                } else {
                    let factor = (distance_from_coast as usize + COAST_FRINGE - COAST_WIDTH) as f32 / COAST_FRINGE as f32 / 2.0;
                    rainfall = get_rainfall(rain_class) * factor + get_rainfall(rain_class + 1) * (1.0 - factor);
                }
                
            }

            // As above, ignore y-deviation.
            if distance_from_river_y == 0 {
                elevation = elevation.min(adj_north_desert_max_elevation * 1.01);
            }
            
            hex_map[y as usize].push(Hex {
                coordinate: (x, y),
                elevation,
                water_depth: 0.0,
                suspended_load: 0.0,
                // TODO: Is the added randomness actually helping? Probably doens't hurt at least.
                rainfall: rainfall * RAINFALL_FACTOR * rng.gen_range(0.9..1.1),
                erosion_multiplier: rng.gen_range(0.95..1.05),
                original_land: elevation > SEA_LEVEL,
            });
        }
    }

    let_slopes_settle(&mut hex_map);
    fill_sea(&mut hex_map);
    // Note: prefilling basins doesn't play so well with evaporation,
    // hence why it's currently commented out.
    // prefill_basins(&mut hex_map);

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    let total_steps = (WIDTH_HEXAGONS as u32) * rounds;
    let final_sea_level = simulate_erosion(&mut hex_map, total_steps, river_outlet_x);

    // TODO: This isn't working, should fix.
    // Count final blue pixels for quick sanity check
    let final_blue = frame_buffer
        .iter()
        .filter(|&&px| (px & 0x0000FF) == 0x0000FF && (px >> 16 & 0xFF) == 0 && (px >> 8 & 0xFF) == 0)
        .count();
    println!("Final blue pixels: {}", final_blue);

    let hex_duration = hex_start.elapsed();
    println!("Hex map creation took: {:?}", hex_duration);
    println!("Seed: {}", seed);

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