use image::RgbImage;
use image::{ImageBuffer, Rgb};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;
mod gpu_simulation;
use gpu_simulation::{GpuSimulation, HexGpu};
use pollster;
mod constants;
use clap::Parser;
use constants::*;
use csv::{ReaderBuilder, WriterBuilder};
use noise::{NoiseFn, Simplex};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;

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

    /// Resume simulation from a CSV save file
    #[arg(long)]
    resume: Option<String>,
}

// Neighbor offsets for "columns-lined" hex layout
const NEIGH_OFFSETS_EVEN: [(i16, i16); 6] = [
    (1, 0),   // 4 o'clock
    (0, 1),   // 6 o'clock
    (-1, 0),  // 8 o'clock
    (0, -1),  // 12 o'clock
    (-1, -1), // 10 o'clock
    (1, -1),  // 2 o'clock
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
    elevation: f32,      // Feet
    water_depth: f32,    // Feet of water currently stored in this hex
    suspended_load: f32, // Feet of sediment stored in water column
    rainfall: f32,       // Feet of rainfall added to this hex per step
    erosion_multiplier: f32,
    uplift: f32, // Feet of elevation added to this hex per step
    original_land: bool,
}

// Map's dimensions define what constitutes a "round", since it determines how long it takes
// for changes on one side of the map to propagate to the other. Using is a function is a
// hack to get around not being able to use max to declare a constant.
fn get_round_size() -> usize {
    WIDTH_HEXAGONS.max(HEIGHT_PIXELS)
}

// This can be done more cleanly with floor division, but the
// docs describe div_floor as part of an experimental API.
fn hex_distance(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    let dx = (x1 - x2).abs();
    if dx % 2 == 0 {
        let diagonal_only_y_min = y1 - dx / 2;
        let diagonal_only_y_max = y1 + dx / 2;
        return dx
            + 0.max(diagonal_only_y_min - y2)
                .max(y2 - diagonal_only_y_max);
    } else {
        let diagonal_only_y_min = y1 - (dx - x1 % 2) / 2;
        let diagonal_only_y_max = y1 + (dx + x1 % 2) / 2;
        return dx
            + 0.max(diagonal_only_y_min - y2)
                .max(y2 - diagonal_only_y_max);
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
    if elevation < BASE_SEA_LEVEL {
        let normalized_elevation =
            ((elevation - ABYSSAL_PLAINS_MAX_DEPTH) / (-1.0 * ABYSSAL_PLAINS_MAX_DEPTH)).min(1.0);
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
            // Yellow (255,255,0) to Orange (255,127,0)
            let factor = (normalized_height - 0.15) / 0.15; // 0..1
            let red = 255;
            let green = (127.0 + 128.0 * (1.0 - factor)) as u8; // 255→127
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.5 {
            // Orange (255,127,0) to Red (255,0,0)
            let factor = (normalized_height - 0.3) / 0.2; // 0..1
            let red = 255;
            let green = (127.0 * (1.0 - factor)) as u8; // 127→0
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.7 {
            // Red (255,0,0) to brown (150,75,0)
            let factor = (normalized_height - 0.5) / 0.2; // 0..1
            let red = 150 + (105.0 * (1.0 - factor)) as u8; // 255→150
            let green = 75 * factor as u8;
            let blue = 0;
            Rgb([red, green, blue])
        } else {
            // Brown (150,75,0) to white (255,255,255)
            let factor = ((normalized_height - 0.7) / 0.3).min(1.0); // 0..1
            let red = 150 + (105.0 * factor) as u8; // 150→255
            let green = 75 + (180.0 * factor) as u8; // 75→255
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
                uplift: h.uplift,
                residual_water_depth: 0.0,
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
        cell.water_depth = h.water_depth + h.residual_water_depth;
        cell.suspended_load = h.suspended_load;
        cell.erosion_multiplier = h.erosion_multiplier;
        cell.uplift = h.uplift;
    }
}

fn let_slopes_settle(hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);
    gpu_sim.resize_min_buffers(width, height);

    upload_hex_data(hex_map, &gpu_sim);

    for _ in 0..200 {
        // This is the only place this function is called,
        // may create opportunities for refactoring.
        gpu_sim.run_repose_step(width, height);
    }

    download_hex_data(&gpu_sim, hex_map);
}

/// Calculates which hexes are "continental" based on connectivity to the eastern land boundary.
/// Continental hexes are:
/// 1. All hexes at x = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH (the eastern boundary)
/// 2. Any hex with elevation > sea_level adjacent to a continental hex (spreading westward)
/// 3. Any enclosed region completely surrounded by continental hexes (inland lakes/valleys)
///
/// This excludes:
/// - Islands (not connected to the continent)
/// - The NE basin (x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH)
/// - Coastal bays connected to the sea
fn calculate_continental_hexes(hex_map: &Vec<Vec<Hex>>, sea_level: f32) -> Vec<Vec<bool>> {
    let height = HEIGHT_PIXELS;
    let width = WIDTH_HEXAGONS;
    let continental_boundary_x = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH;

    let mut continental = vec![vec![false; width]; height];
    let mut queue = std::collections::VecDeque::new();

    // Step 1: Initialize hexes at the continental boundary as continental
    for y in 0..height {
        continental[y][continental_boundary_x] = true;
        queue.push_back((continental_boundary_x, y));
    }

    // Step 2: BFS to spread continental status to adjacent hexes above sea level
    while let Some((x, y)) = queue.pop_front() {
        let offsets = if x % 2 == 0 {
            &NEIGH_OFFSETS_EVEN
        } else {
            &NEIGH_OFFSETS_ODD
        };

        for &(dx, dy) in offsets.iter() {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;

            if nx >= 0 && nx < width as i16 && ny >= 0 && ny < height as i16 {
                let nx = nx as usize;
                let ny = ny as usize;

                // Skip if already continental or if x > continental_boundary_x
                if continental[ny][nx] || nx > continental_boundary_x {
                    continue;
                }

                // Only spread to hexes above sea level
                if hex_map[ny][nx].elevation > sea_level {
                    continental[ny][nx] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    // Step 3: Find hexes reachable from ocean edges (through non-continental hexes)
    let mut sea_reachable = vec![vec![false; width]; height];

    // Start from western edge (x=0) and southern edge (y=height-1)
    for y in 0..height {
        if !continental[y][0] {
            sea_reachable[y][0] = true;
            queue.push_back((0, y));
        }
    }
    for x in 0..width {
        if !continental[height - 1][x] && !sea_reachable[height - 1][x] {
            sea_reachable[height - 1][x] = true;
            queue.push_back((x, height - 1));
        }
    }

    // BFS through non-continental hexes to find all sea-reachable areas
    while let Some((x, y)) = queue.pop_front() {
        let offsets = if x % 2 == 0 {
            &NEIGH_OFFSETS_EVEN
        } else {
            &NEIGH_OFFSETS_ODD
        };

        for &(dx, dy) in offsets.iter() {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;

            if nx >= 0 && nx < width as i16 && ny >= 0 && ny < height as i16 {
                let nx = nx as usize;
                let ny = ny as usize;

                if !continental[ny][nx] && !sea_reachable[ny][nx] {
                    sea_reachable[ny][nx] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    // Step 4: Any hex not continental AND not sea-reachable is an enclosed inland area
    // Mark these as continental (they're surrounded by continental land)
    for y in 0..height {
        for x in 0..width {
            if !continental[y][x] && !sea_reachable[y][x] {
                continental[y][x] = true;
            }
        }
    }

    continental
}

// TODO: Is it possible to simplify this?
fn fill_sea(hex_map: &mut Vec<Vec<Hex>>) {
    let height = HEIGHT_PIXELS as usize;
    let width = WIDTH_HEXAGONS as usize;

    // Track which hexes are reachable from the edges
    let mut reachable = vec![vec![false; width]; height];
    let mut queue = std::collections::VecDeque::new();

    // Start from western edge (x=0) and southern edge (y=HEIGHT_PIXELS-1)
    // Add all hexes on these edges with elevation < 0 to the queue
    for y in 0..height {
        let x = 0;
        if hex_map[y][x].elevation < 0.0 {
            reachable[y][x] = true;
            queue.push_back((x, y));
        }
    }
    for x in 0..width {
        let y = height - 1;
        if hex_map[y][x].elevation < 0.0 {
            if !reachable[y][x] {
                reachable[y][x] = true;
                queue.push_back((x, y));
            }
        }
    }

    // BFS: explore all hexes reachable from the edges via paths with elevation < 0
    while let Some((x, y)) = queue.pop_front() {
        let offsets = if x % 2 == 0 {
            &NEIGH_OFFSETS_EVEN
        } else {
            &NEIGH_OFFSETS_ODD
        };

        for &(dx, dy) in offsets.iter() {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;

            if nx >= 0 && nx < width as i16 && ny >= 0 && ny < height as i16 {
                let nx = nx as usize;
                let ny = ny as usize;

                if !reachable[ny][nx] && hex_map[ny][nx].elevation < 0.0 {
                    reachable[ny][nx] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    // Fill all reachable hexes with water
    for y in 0..height {
        for x in 0..width {
            if reachable[y][x] {
                let cell = &mut hex_map[y][x];
                cell.water_depth = BASE_SEA_LEVEL - cell.elevation;
            }
        }
    }
}

fn simulate_erosion(
    hex_map: &mut Vec<Vec<Hex>>,
    steps: u32,
    seed: u32,
    starting_step: u32,
    initial_max_elevation: f32,
    initial_avg_elevation: f32,
    initial_sea_avg_elevation: f32,
    initial_north_avg: f32,
    initial_central_avg: f32,
    initial_south_avg: f32,
    prior_elapsed_secs: f64,
) -> (f32, f64) {
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

    let mut current_sea_level = BASE_SEA_LEVEL;

    upload_hex_data(hex_map, &gpu_sim);

    println!(
        "Calculated constants: NORTH_DESERT_WIDTH {}  NE_BASIN_WIDTH {}  TOTAL_LAND_WIDTH {}  TOTAL_SEA_WIDTH {}",
        NORTH_DESERT_WIDTH, NE_BASIN_WIDTH, TOTAL_LAND_WIDTH, TOTAL_SEA_WIDTH
    );
    println!(
        "  BASE_SEA_LEVEL {}  NORTH_DESERT_HEIGHT {}  CENTRAL_HIGHLAND_HEIGHT {}  SOUTH_MOUNTAINS_HEIGHT {}",
        BASE_SEA_LEVEL, NORTH_DESERT_HEIGHT, CENTRAL_HIGHLAND_HEIGHT, SOUTH_MOUNTAINS_HEIGHT
    );
    println!(
        "  RAINFALL_FACTOR {}  EVAPORATION_FACTOR {}",
        RAINFALL_FACTOR, EVAPORATION_FACTOR
    );
    println!("  RIVER_Y {}  RIVER_SOURCE_X {}", RIVER_Y, RIVER_SOURCE_X);

    let round_size = get_round_size();

    // TODO: Currently dividing by 1000 to compensate for heartbeat not working, really need to fix that.
    let log_steps = round_size as u32 * LOG_ROUNDS / 100;

    for step_offset in 0..steps {
        let step = starting_step + step_offset;
        let seasonal_rain_multiplier = 1.0
            + (step as f32 * 2.0 * std::f32::consts::PI * YEARS_PER_STEP + std::f32::consts::PI)
                .cos();

        if step % log_steps == 0 {
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
                .map(|row| {
                    row.iter()
                        .filter(|h| h.elevation > current_sea_level)
                        .fold(0.0, |acc, h| acc + h.rainfall)
                })
                .sum();

            let cells_above_sea_level: usize = hex_map
                .par_iter()
                .map(|row| {
                    row.iter()
                        .filter(|h| h.elevation > current_sea_level)
                        .count()
                })
                .sum();

            let (water_on_land, max_depth) = hex_map
                .par_iter()
                .map(|row| {
                    let mut sum = 0.0f64;
                    let mut row_max = 0.0f64;
                    for h in row {
                        if h.elevation > current_sea_level {
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
                    |acc, val| (acc.0 + val.0, acc.1.max(val.1)),
                );

            let mean_depth = water_on_land / cells_above_sea_level as f64;

            let wet_cells: usize = hex_map
                .par_iter()
                .map(|row| {
                    row.iter()
                        .filter(|h| {
                            h.elevation > current_sea_level && h.water_depth > LOW_WATER_THRESHOLD
                        })
                        .count()
                })
                .sum();

            let wet_cells_percentage = wet_cells as f64 / cells_above_sea_level as f64 * 100.0;

            // Find source_y as the y of hex with greatest water depth where y < NORTH_DESERT_HEIGHT and x = RIVER_SOURCE_X
            let source_y: usize = hex_map
                .par_iter()
                .enumerate()
                .take(NORTH_DESERT_HEIGHT)
                .filter_map(|(y, row)| {
                    row.get(RIVER_SOURCE_X)
                        .filter(|hex| {
                            hex.elevation > current_sea_level
                                && hex.water_depth > LOW_WATER_THRESHOLD
                        })
                        .map(|hex| (y, hex.water_depth))
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(y, _)| y)
                .unwrap_or(RIVER_Y);

            let source_hex = &hex_map[source_y][RIVER_SOURCE_X];
            let outlet_hex = &hex_map[RIVER_Y][TOTAL_SEA_WIDTH];
            let target_delta_hex = &hex_map[RIVER_Y][TOTAL_SEA_WIDTH - 231];

            // The more the hex we want to be the edge of the delta is dry, the more we want
            // we want to bring the sea level up to match.
            // TODO: Is there an inexpensive way to download this every step?
            // if target_delta_hex.elevation - target_delta_hex.water_depth > current_sea_level {
            //     current_sea_level = target_delta_hex.elevation - target_delta_hex.water_depth;
            // }

            let round = step / (round_size as u32);

            // Calculate continental hexes for this logging step
            let continental = calculate_continental_hexes(hex_map, current_sea_level);

            // Regional boundaries based on y coordinate
            let central_boundary = NORTH_DESERT_HEIGHT;
            let south_boundary = NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT;

            // Calculate overall and regional stats for continental hexes
            let (
                min_elevation,
                max_elevation,
                total_land,
                continental_count,
                north_total,
                north_count,
                central_total,
                central_count_region,
                south_total,
                south_count,
            ): (f32, f32, f64, usize, f64, usize, f64, usize, f64, usize) = {
                let mut min_elev = f32::INFINITY;
                let mut max_elev = f32::NEG_INFINITY;
                let mut sum = 0.0f64;
                let mut count = 0usize;
                let mut n_sum = 0.0f64;
                let mut n_count = 0usize;
                let mut c_sum = 0.0f64;
                let mut c_count = 0usize;
                let mut s_sum = 0.0f64;
                let mut s_count = 0usize;

                for y in 0..HEIGHT_PIXELS {
                    for x in 0..WIDTH_HEXAGONS {
                        if continental[y][x] {
                            let elev = hex_map[y][x].elevation;
                            let elev_above_sea = (elev - current_sea_level) as f64;

                            if elev < min_elev {
                                min_elev = elev;
                            }
                            if elev > max_elev {
                                max_elev = elev;
                            }
                            sum += elev_above_sea;
                            count += 1;

                            // Regional breakdown
                            if y < central_boundary {
                                n_sum += elev_above_sea;
                                n_count += 1;
                            } else if y < south_boundary {
                                c_sum += elev_above_sea;
                                c_count += 1;
                            } else {
                                s_sum += elev_above_sea;
                                s_count += 1;
                            }
                        }
                    }
                }
                (
                    min_elev, max_elev, sum, count, n_sum, n_count, c_sum, c_count, s_sum, s_count,
                )
            };

            let avg_elevation = if continental_count > 0 {
                (total_land / continental_count as f64) as f32
            } else {
                0.0
            };

            let north_avg = if north_count > 0 {
                (north_total / north_count as f64) as f32
            } else {
                0.0
            };
            let central_avg = if central_count_region > 0 {
                (central_total / central_count_region as f64) as f32
            } else {
                0.0
            };
            let south_avg = if south_count > 0 {
                (south_total / south_count as f64) as f32
            } else {
                0.0
            };

            // Calculate sea hex average elevation
            // Sea hexes: x <= TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH, not continental, elevation < sea level
            let (sea_total, sea_count): (f64, usize) = {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                let sea_boundary_x = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH;

                for y in 0..HEIGHT_PIXELS {
                    for x in 0..=sea_boundary_x {
                        if !continental[y][x] && hex_map[y][x].elevation < current_sea_level {
                            sum += hex_map[y][x].elevation as f64;
                            count += 1;
                        }
                    }
                }
                (sum, count)
            };

            let sea_avg_elevation = if sea_count > 0 {
                (sea_total / sea_count as f64) as f32
            } else {
                0.0
            };

            let net_land: f64 = hex_map
                .iter()
                .flat_map(|row| row.iter())
                .map(|h| (h.elevation - current_sea_level) as f64)
                .sum();

            let total_sediment: f64 = hex_map
                .iter()
                .flat_map(|row| row.iter())
                .map(|h| h.suspended_load as f64)
                .sum();

            let total_elapsed_secs = prior_elapsed_secs + water_start.elapsed().as_secs_f64();
            let total_elapsed = std::time::Duration::from_secs_f64(total_elapsed_secs);

            // TODO(topher): Add tenths of rounds.
            println!(
                "Seed {}, round {:.0}.{:.0}:",
                seed,
                round,
                step % (round_size as u32) * 10 / (round_size as u32),
            );
            println!(
                "  water in {:.3}  stored {:.3}  mean depth {:.3} ft  max depth {:.3} ft  wet {:} ({:.1}%)",
                rainfall_added,
                water_on_land,
                mean_depth,
                max_depth,
                wet_cells,
                wet_cells_percentage
            );
            println!(
                "  sea level: {:.3} ft  seasonal rain multiplier: {:.3}",
                current_sea_level, seasonal_rain_multiplier
            );
            println!(
                "  source elevation {:.3} ft  source water depth {:.3} ft  source sediment {:.3} ft  outlet elevation {:.3} ft  target delta elevation {:.3} ft",
                source_hex.elevation - current_sea_level,
                source_hex.water_depth,
                source_hex.suspended_load,
                outlet_hex.elevation - current_sea_level,
                target_delta_hex.elevation - current_sea_level,
            );
            println!(
                "  total land (original): {:.3} ft  net land: {:.3} ft  total sediment: {:.3} ft",
                total_land, net_land, total_sediment
            );
            println!(
                "  max elevation: {:.3} ft (initial: {:.3} ft, {:.1}%)  avg elevation: {:.3} ft (initial: {:.3} ft, {:.1}%)",
                max_elevation - current_sea_level,
                initial_max_elevation,
                (max_elevation - current_sea_level) / initial_max_elevation * 100.0,
                avg_elevation,
                initial_avg_elevation,
                avg_elevation / initial_avg_elevation * 100.0,
            );
            println!(
                "  regional avg: north {:.1} ft ({:.1}%)  central {:.1} ft ({:.1}%)  south {:.1} ft ({:.1}%)",
                north_avg, north_avg / initial_north_avg * 100.0,
                central_avg, central_avg / initial_central_avg * 100.0,
                south_avg, south_avg / initial_south_avg * 100.0,
            );
            println!(
                "  min elevation: {:.3} ft  sea avg: {:.3} ft (initial: {:.3} ft, {:.1}%)  time: {:?}",
                min_elevation - current_sea_level,
                sea_avg_elevation,
                initial_sea_avg_elevation,
                sea_avg_elevation / initial_sea_avg_elevation * 100.0,
                total_elapsed,
            );

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
            save_buffer_png(
                "terrain_water.png",
                &frame_buffer,
                WIDTH_PIXELS as u32,
                HEIGHT_PIXELS as u32,
            );

            render_frame(hex_map, &mut frame_buffer, current_sea_level, false);

            let mut tag = "";
            if step == 0 {
                tag = "_initial";
            }
            save_buffer_png(
                &format!("terrain{tag}.png", tag = tag),
                &frame_buffer,
                WIDTH_PIXELS as u32,
                HEIGHT_PIXELS as u32,
            );
            // Only save CSV every 100 rounds to avoid performance issues
            if step % (1000 * log_steps) == 0 {
                let total_elapsed_secs = prior_elapsed_secs + water_start.elapsed().as_secs_f64();
                save_simulation_state_csv(
                    &format!("terrain{tag}.csv", tag = tag),
                    hex_map,
                    seed,
                    step,
                    initial_max_elevation,
                    initial_avg_elevation,
                    initial_sea_avg_elevation,
                    initial_north_avg,
                    initial_central_avg,
                    initial_south_avg,
                    total_elapsed_secs,
                );
            }
        } else if step % (log_steps / 10) == 0 {
            // TODO: Get the heartbeat to work.
            // gpu_sim.heartbeat();
            gpu_sim.download_hex_data();
        }

        // let tidal_adjustment: f32 = 3.0 * (2.0 * f32::PI * step as f32 / TIDE_INTERVAL_STEPS as f32).sin();
        let tidal_adjustment: f32 = 0.0;

        gpu_sim.run_simulation_step_batched(
            width,
            height,
            current_sea_level + tidal_adjustment,
            seasonal_rain_multiplier,
        );
        current_sea_level = BASE_SEA_LEVEL + 0.02 * YEARS_PER_STEP * step as f32;
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

    let westernmost_land_hex = hex_map
        .par_iter()
        .map(|row| {
            row.iter()
                .filter(|h| h.water_depth <= LOW_WATER_THRESHOLD)
                .min_by_key(|h| h.coordinate.0)
        })
        .flatten()
        .min_by_key(|h| h.coordinate.0);

    let westernmost_land_hex_north = hex_map
        .par_iter()
        .take(NORTH_DESERT_HEIGHT)
        .map(|row| {
            row.iter()
                .filter(|h| h.water_depth <= LOW_WATER_THRESHOLD)
                .min_by_key(|h| h.coordinate.0)
        })
        .flatten()
        .min_by_key(|h| h.coordinate.0);

    let westernmost_land_hex_central = hex_map
        .par_iter()
        .skip(NORTH_DESERT_HEIGHT)
        .take(CENTRAL_HIGHLAND_HEIGHT)
        .map(|row| {
            row.iter()
                .filter(|h| h.water_depth <= LOW_WATER_THRESHOLD)
                .min_by_key(|h| h.coordinate.0)
        })
        .flatten()
        .min_by_key(|h| h.coordinate.0);

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft-hexes,",
        steps, total_outflow
    );

    println!(
        " westernmost land: {}, westernmost land north: {}, westernmost land south: {},",
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

    let total_elapsed_secs = prior_elapsed_secs + water_start.elapsed().as_secs_f64();
    (current_sea_level, total_elapsed_secs)
}

fn print_elevation_and_sediment(gpu_sim: &GpuSimulation, step_label: &str) {
    let hex_data = gpu_sim.download_hex_data();
    let sum_elev: f64 = hex_data.iter().map(|h| h.elevation as f64).sum();
    let sum_sed: f64 = hex_data.iter().map(|h| h.suspended_load as f64).sum();
    println!(
        "{}: sum_elev {:.6}  sum_sed {:.6}  sum_mass {:.6}",
        step_label,
        sum_elev,
        sum_sed,
        sum_elev + sum_sed
    );
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

fn save_simulation_state_csv(
    path: &str,
    hex_map: &Vec<Vec<Hex>>,
    seed: u32,
    step: u32,
    initial_max_elevation: f32,
    initial_avg_elevation: f32,
    initial_sea_avg_elevation: f32,
    initial_north_avg: f32,
    initial_central_avg: f32,
    initial_south_avg: f32,
    elapsed_secs: f64,
) {
    let file = File::create(path).expect("Failed to create CSV file");
    let mut wtr = WriterBuilder::new().flexible(true).from_writer(file);

    // Write mini-table for seed, step, initial elevations, and elapsed time at the top
    wtr.write_record(&[
        "seed",
        "step",
        "initial_max_elevation",
        "initial_avg_elevation",
        "initial_sea_avg_elevation",
        "initial_north_avg",
        "initial_central_avg",
        "initial_south_avg",
        "elapsed_secs",
    ])
    .expect("Failed to write metadata header");
    wtr.write_record(&[
        seed.to_string(),
        step.to_string(),
        initial_max_elevation.to_bits().to_string(),
        initial_avg_elevation.to_bits().to_string(),
        initial_sea_avg_elevation.to_bits().to_string(),
        initial_north_avg.to_bits().to_string(),
        initial_central_avg.to_bits().to_string(),
        initial_south_avg.to_bits().to_string(),
        elapsed_secs.to_bits().to_string(),
    ])
    .expect("Failed to write metadata values");

    // Blank separator row
    wtr.write_record(&[""])
        .expect("Failed to write separator row");

    // Write main hex data header
    wtr.write_record(&[
        "x",
        "y",
        "elevation",
        "water_depth",
        "suspended_load",
        "rainfall",
        "erosion_multiplier",
        "uplift",
        "original_land",
    ])
    .expect("Failed to write CSV header");

    // Write each hex as a row
    for (y, row) in hex_map.iter().enumerate() {
        for (x, hex) in row.iter().enumerate() {
            // Bitcast floating point values to preserve exact precision
            let elevation_bits = hex.elevation.to_bits();
            let water_depth_bits = hex.water_depth.to_bits();
            let suspended_load_bits = hex.suspended_load.to_bits();
            let rainfall_bits = hex.rainfall.to_bits();
            let erosion_multiplier_bits = hex.erosion_multiplier.to_bits();
            let uplift_bits = hex.uplift.to_bits();

            wtr.write_record(&[
                x.to_string(),
                y.to_string(),
                elevation_bits.to_string(),
                water_depth_bits.to_string(),
                suspended_load_bits.to_string(),
                rainfall_bits.to_string(),
                erosion_multiplier_bits.to_string(),
                uplift_bits.to_string(),
                hex.original_land.to_string(),
            ])
            .expect("Failed to write CSV record");
        }
    }

    wtr.flush().expect("Failed to flush CSV writer");
}

fn load_simulation_state_csv(
    path: &str,
) -> (Vec<Vec<Hex>>, u32, u32, f32, f32, f32, f32, f32, f32, f64) {
    let file = File::open(path).expect("Failed to open CSV file");
    let mut rdr = ReaderBuilder::new().flexible(true).from_reader(file);

    let mut hex_map: Vec<Vec<Hex>> = Vec::new();
    let mut seed = 0u32;
    let mut step = 0u32;
    let mut initial_max_elevation = 0.0f32;
    let mut initial_avg_elevation = 0.0f32;
    let mut initial_sea_avg_elevation = 0.0f32;
    let mut initial_north_avg = 0.0f32;
    let mut initial_central_avg = 0.0f32;
    let mut initial_south_avg = 0.0f32;
    let mut elapsed_secs = 0.0f64;
    let mut row_index = 0usize;

    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        row_index += 1;

        // Row 1: metadata values (seed, step, initial elevations, elapsed_secs) - header was consumed by Reader
        if row_index == 1 {
            seed = record[0].parse().expect("Failed to parse seed");
            step = record[1].parse().expect("Failed to parse step");
            // Handle backward compatibility - older saves may not have these fields
            if record.len() > 2 {
                let initial_max_bits: u32 = record[2]
                    .parse()
                    .expect("Failed to parse initial_max_elevation bits");
                initial_max_elevation = f32::from_bits(initial_max_bits);
            }
            if record.len() > 3 {
                let initial_avg_bits: u32 = record[3]
                    .parse()
                    .expect("Failed to parse initial_avg_elevation bits");
                initial_avg_elevation = f32::from_bits(initial_avg_bits);
            }
            if record.len() > 4 {
                let initial_sea_avg_bits: u32 = record[4]
                    .parse()
                    .expect("Failed to parse initial_sea_avg_elevation bits");
                initial_sea_avg_elevation = f32::from_bits(initial_sea_avg_bits);
            }
            if record.len() > 5 {
                let initial_north_bits: u32 = record[5]
                    .parse()
                    .expect("Failed to parse initial_north_avg bits");
                initial_north_avg = f32::from_bits(initial_north_bits);
            }
            if record.len() > 6 {
                let initial_central_bits: u32 = record[6]
                    .parse()
                    .expect("Failed to parse initial_central_avg bits");
                initial_central_avg = f32::from_bits(initial_central_bits);
            }
            if record.len() > 7 {
                let initial_south_bits: u32 = record[7]
                    .parse()
                    .expect("Failed to parse initial_south_avg bits");
                initial_south_avg = f32::from_bits(initial_south_bits);
            }
            if record.len() > 8 {
                let elapsed_bits: u64 = record[8]
                    .parse()
                    .expect("Failed to parse elapsed_secs bits");
                elapsed_secs = f64::from_bits(elapsed_bits);
            }
            continue;
        }

        // Row 2: blank separator - skip
        if row_index == 2 {
            continue;
        }

        // Row 3: hex data header - skip
        if row_index == 3 {
            continue;
        }

        // Row 4+: hex data
        let x: usize = record[0].parse().expect("Failed to parse x coordinate");
        let y: usize = record[1].parse().expect("Failed to parse y coordinate");

        // Ensure we have enough rows
        while hex_map.len() <= y {
            hex_map.push(Vec::new());
        }

        // Ensure we have enough columns in this row
        while hex_map[y].len() <= x {
            let current_x = hex_map[y].len();
            hex_map[y].push(Hex {
                coordinate: (current_x, y),
                elevation: 0.0,
                water_depth: 0.0,
                suspended_load: 0.0,
                rainfall: 0.0,
                erosion_multiplier: 1.0,
                uplift: 0.0,
                original_land: false,
            });
        }

        // Convert bitcast values back to floats
        let elevation_bits: u32 = record[2].parse().expect("Failed to parse elevation bits");
        let water_depth_bits: u32 = record[3].parse().expect("Failed to parse water_depth bits");
        let suspended_load_bits: u32 = record[4]
            .parse()
            .expect("Failed to parse suspended_load bits");
        let rainfall_bits: u32 = record[5].parse().expect("Failed to parse rainfall bits");
        let erosion_multiplier_bits: u32 = record[6]
            .parse()
            .expect("Failed to parse erosion_multiplier bits");
        let uplift_bits: u32 = record[7].parse().expect("Failed to parse uplift bits");
        let original_land: bool = record[8].parse().expect("Failed to parse original_land");

        hex_map[y][x] = Hex {
            coordinate: (x, y),
            elevation: f32::from_bits(elevation_bits),
            water_depth: f32::from_bits(water_depth_bits),
            suspended_load: f32::from_bits(suspended_load_bits),
            rainfall: f32::from_bits(rainfall_bits),
            erosion_multiplier: f32::from_bits(erosion_multiplier_bits),
            uplift: f32::from_bits(uplift_bits),
            original_land,
        };
    }

    (
        hex_map,
        seed,
        step,
        initial_max_elevation,
        initial_avg_elevation,
        initial_sea_avg_elevation,
        initial_north_avg,
        initial_central_avg,
        initial_south_avg,
        elapsed_secs,
    )
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
            let color = if show_water && hex.water_depth > LOW_WATER_THRESHOLD {
                let r = 0u8;
                let g =
                    (255.0 * 0.4 * (1.0 - hex.water_depth / HIGH_WATER_THRESHOLD)).max(0.0) as u8;
                let b =
                    (255.0 * (0.4 + 0.6 * hex.water_depth / HIGH_WATER_THRESHOLD)).max(0.0) as u8;
                (r as u32) << 16 | (g as u32) << 8 | (b as u32)
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

// Renders rainfall map - uses a blue gradient where darker = more rain
// max_rainfall is the value that maps to full intensity (expected ~95.8 * RAINFALL_FACTOR)
fn render_rainfall(hex_map: &Vec<Vec<Hex>>, buffer: &mut [u32], max_rainfall: f32) {
    let mut actual_max: f32 = 0.0;
    let mut actual_min: f32 = f32::MAX;

    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            let hex_x = ((x as f32) / HEX_FACTOR) as usize;
            let hex_y = y;

            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);

            let hex = &hex_map[hex_y as usize][hex_x as usize];
            let rainfall = hex.rainfall;

            if rainfall > actual_max {
                actual_max = rainfall;
            }
            if rainfall > 0.0 && rainfall < actual_min {
                actual_min = rainfall;
            }

            // Normalize rainfall to 0-1 range
            let normalized = (rainfall / max_rainfall).clamp(0.0, 1.0);

            // Use a gradient from light cyan (low rain) to dark blue (high rain)
            // For zero rainfall, use tan/beige (desert)
            // Note: rainfall values are tiny due to RAINFALL_FACTOR, so compare against 0.0
            let color = if rainfall == 0.0 {
                // Desert - tan color
                let r = 210u8;
                let g = 180u8;
                let b = 140u8;
                (r as u32) << 16 | (g as u32) << 8 | (b as u32)
            } else {
                // Rain gradient: light cyan to dark blue
                let r = (200.0 * (1.0 - normalized)) as u8;
                let g = (220.0 * (1.0 - normalized * 0.7)) as u8;
                let b = (100.0 + 155.0 * normalized) as u8;
                (r as u32) << 16 | (g as u32) << 8 | (b as u32)
            };
            buffer[(y as usize) * (WIDTH_PIXELS as usize) + (x as usize)] = color;
        }
    }

    // Divide by RAINFALL_FACTOR to show human-readable values (annual inches equivalent)
    println!(
        "Rainfall stats (÷RAINFALL_FACTOR): min={:.2}, max={:.2}, expected_max={:.2}",
        actual_min / RAINFALL_FACTOR,
        actual_max / RAINFALL_FACTOR,
        max_rainfall / RAINFALL_FACTOR
    );
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
fn get_simplex_noise(simplex: &Simplex, input: f64, period: f64) -> f32 {
    (simplex.get([input / period, 0.0]) as f32 + 1.0) / 2.0
}

// Generates a value between 0 and 1.
fn get_simplex_noise_for_hex(simplex: &Simplex, x: f64, y: f64, period: f64) -> f32 {
    let adjusted_y = y + (x % 2.0) * 0.5;
    (simplex.get([x * HEX_FACTOR as f64 / period, adjusted_y / period]) as f32 + 1.0) / 2.0
}

fn get_simplex_noise_map(simplex: &Simplex) -> Vec<Vec<f32>> {
    let mut noise_map = Vec::new();
    for y in 0..HEIGHT_PIXELS {
        let mut row = Vec::new();
        for x in 0..WIDTH_HEXAGONS {
            let map_third_noise = get_simplex_noise_for_hex(
                &simplex,
                x as f64,
                y as f64,
                NORTH_DESERT_HEIGHT
                    .min(CENTRAL_HIGHLAND_HEIGHT)
                    .min(SOUTH_MOUNTAINS_HEIGHT) as f64
                    - TRANSITION_PERIOD as f64,
            );
            let transition_period_noise = get_simplex_noise_for_hex(
                &simplex,
                (x + WIDTH_HEXAGONS) as f64,
                y as f64,
                TRANSITION_PERIOD as f64,
            );
            let coastal_noise = get_simplex_noise_for_hex(
                &simplex,
                (x + WIDTH_HEXAGONS * 2) as f64,
                y as f64,
                COAST_WIDTH as f64,
            );
            let deviation_noise = get_simplex_noise_for_hex(
                &simplex,
                (x + WIDTH_HEXAGONS * 3) as f64,
                y as f64,
                DEVIATION_PERIOD as f64,
            );
            let simplex_noise =
                ((transition_period_noise + coastal_noise + map_third_noise + deviation_noise)
                    / 4.0)
                    .powf(3.0_f32.sqrt());

            if simplex_noise < 0.0 || simplex_noise > 1.0 {
                println!("Simplex noise out of range: {}", simplex_noise);
                println!(
                    "x: {}, y: {}, TRANSITION_PERIOD: {}, COAST_WIDTH: {}",
                    x, y, TRANSITION_PERIOD, COAST_WIDTH
                );
                println!(
                    "map_third_noise: {}, transition_period_noise: {}, coastal_noise: {}",
                    map_third_noise, transition_period_noise, coastal_noise
                );
                panic!("Simplex noise out of range");
            }

            row.push(simplex_noise);
        }
        noise_map.push(row);
    }
    noise_map
}

fn get_normalized_simplex_noise_map(simplex: &Simplex) -> Vec<Vec<f32>> {
    let noise_map = get_simplex_noise_map(simplex);
    let min_noise = noise_map
        .iter()
        .flatten()
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_noise = noise_map
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_noise - min_noise;
    noise_map
        .iter()
        .map(|row| {
            row.iter()
                .map(|noise| (noise - min_noise) / range)
                .collect()
        })
        .collect()
}

// TODO: Library function for this?
fn get_white_noise(seed: u32, x: usize, y: usize) -> f32 {
    let hex_seed = seed
        .wrapping_add((x as u32).wrapping_mul(7919))
        .wrapping_add((y as u32).wrapping_mul(982451653));
    let mut rng = StdRng::seed_from_u64(hex_seed as u64);
    rng.gen_range(0.0..1.0)
}

// Generates a value between -COAST_WIDTH/2 and COAST_WIDTH/2.
fn get_sea_deviation(simplex: &Simplex, y: f64, period: f64) -> i16 {
    ((get_simplex_noise(simplex, y + HEIGHT_PIXELS as f64, period) - 0.5) * COAST_WIDTH as f32)
        as i16
}

// Generates a value between -12 and 12.
fn get_land_deviation(simplex: &Simplex, x: f64, y: f64, period: f64) -> i16 {
    ((get_simplex_noise_for_hex(simplex, x - WIDTH_HEXAGONS as f64, y, period) - 0.5) * 24.0 * 2.0)
        as i16
}

// fn get_rainfall(rain_class: i32) -> f32 {
//     match rain_class {
//         0 => VERY_LOW_RAIN,
//         1 => LOW_RAIN,
//         2 => MEDIUM_RAIN,
//         3 => HIGH_RAIN,
//         4 => VERY_HIGH_RAIN,
//         _ => 0.0,
//     }
// }

fn get_rainfall(y: usize, distance_from_coast: f32) -> f32 {
    let mut result = 0.0;
    let latitude = 25.0 + (y as f32 / 2.0 / ONE_DEGREE_LATITUDE_MILES);
    if latitude < 29.0 {
        result = 1.0 / (30.0 - latitude);
    } else if latitude < 36.0 {
        result = latitude - 28.0;
    } else {
        result = (latitude - 35.0).powf(2.0) + 7.0;
    }

    result = result.clamp(0.0, 20.0);

    let factor = ((144.0 / HEX_FACTOR as f32 - distance_from_coast) / (144.0 / HEX_FACTOR as f32))
        .clamp(0.0, 1.0);
    result *= 1.0 + factor * 2.0;
    result
}

fn pick_value_from_range(normal: f32, min: f32, max: f32) -> f32 {
    if normal < 0.0 || normal > 1.0 {
        println!("Normal out of range: {}", normal);
        panic!("Normal out of range");
    }
    if min > max {
        println!("Min is greater than max: {} > {}", min, max);
        panic!("Min is greater than max");
    }
    (normal * (max - min) + min)
}

fn get_boundary_factor(base_factor: f32) -> f32 {
    let x = base_factor.clamp(0.0, 1.0);
    3.0 * x.powf(2.0) - 2.0 * x.powf(3.0)
    // base_factor.clamp(0.0, 1.0)
}

fn main() {
    let args = Args::parse();

    // Validate that seed and resume aren't both provided
    if args.seed.is_some() && args.resume.is_some() {
        eprintln!("Error: Cannot specify both --seed and --resume flags");
        std::process::exit(1);
    }

    let rounds = args.rounds;

    let (
        mut hex_map,
        seed,
        starting_step,
        initial_max_elevation,
        initial_avg_elevation,
        initial_sea_avg_elevation,
        initial_north_avg,
        initial_central_avg,
        initial_south_avg,
        prior_elapsed_secs,
    ) = if let Some(resume_path) = args.resume {
        // Resume from save file
        println!("Resuming simulation from: {}", resume_path);
        let (
            loaded_hex_map,
            loaded_seed,
            loaded_step,
            loaded_initial_max,
            loaded_initial_avg,
            loaded_initial_sea_avg,
            loaded_initial_north,
            loaded_initial_central,
            loaded_initial_south,
            loaded_elapsed_secs,
        ) = load_simulation_state_csv(&resume_path);
        println!(
                "Loaded seed: {}, starting step: {}, initial max elevation: {:.3}, initial avg elevation: {:.3}, initial sea avg: {:.3}, prior elapsed: {:.1}s",
                loaded_seed, loaded_step, loaded_initial_max, loaded_initial_avg, loaded_initial_sea_avg, loaded_elapsed_secs
            );
        println!(
            "  regional initial avgs: north {:.3}, central {:.3}, south {:.3}",
            loaded_initial_north, loaded_initial_central, loaded_initial_south
        );

        let simplex = Simplex::new(loaded_seed);
        let _sea_deviation_for_river_y: i16 =
            get_sea_deviation(&simplex, RIVER_Y as f64, HEIGHT_PIXELS as f64 / 1.5);

        (
            loaded_hex_map,
            loaded_seed,
            loaded_step,
            loaded_initial_max,
            loaded_initial_avg,
            loaded_initial_sea_avg,
            loaded_initial_north,
            loaded_initial_central,
            loaded_initial_south,
            loaded_elapsed_secs,
        )
    } else {
        // New simulation
        let seed = args
            .seed
            .unwrap_or_else(|| rand::thread_rng().gen_range(0..u32::MAX));

        println!("Starting new simulation with seed: {}", seed);

        let mut hex_map = Vec::new();

        let simplex = Simplex::new(seed);
        let noise_map = get_normalized_simplex_noise_map(&simplex);
        let sea_deviation_for_river_y: i16 =
            get_sea_deviation(&simplex, RIVER_Y as f64, HEIGHT_PIXELS as f64 / 1.5);

        let x_deviation_for_outlet = sea_deviation_for_river_y
            + get_land_deviation(
                &simplex,
                TOTAL_SEA_WIDTH as f64,
                RIVER_Y as f64,
                DEVIATION_PERIOD,
            );
        let y_deviation_for_outlet = get_land_deviation(
            &simplex,
            (TOTAL_SEA_WIDTH + WIDTH_HEXAGONS) as f64,
            RIVER_Y as f64,
            96.0,
        );

        // Time hex map creation
        let hex_start = Instant::now();

        for y in 0..HEIGHT_PIXELS {
            hex_map.push(Vec::new());
            let distance_from_river_y = (y as i16 - RIVER_Y as i16).abs();
            let sea_deviation = get_sea_deviation(&simplex, y as f64, HEIGHT_PIXELS as f64 / 1.5);
            // let shelf_deviation = get_sea_deviation(&simplex, (y + HEIGHT_PIXELS * 3 / 2) as f64, HEIGHT_PIXELS as f64);
            let cut_factor = -0.06
                - get_simplex_noise(
                    &simplex,
                    (y + HEIGHT_PIXELS * 5 / 2) as f64,
                    HEIGHT_PIXELS as f64 / 1.5,
                ) * 0.06;

            for x in 0..WIDTH_HEXAGONS {
                let mut elevation = 0.0;
                let mut local_max = SOUTH_MOUNTAINS_MAX_ELEVATION;
                let mut uplift = 0.0;
                let x_deviation = x_deviation_for_outlet
                    - sea_deviation
                    - get_land_deviation(&simplex, x as f64, y as f64, DEVIATION_PERIOD);
                let y_deviation = y_deviation_for_outlet
                    - get_land_deviation(
                        &simplex,
                        (x + WIDTH_HEXAGONS) as f64,
                        y as f64,
                        DEVIATION_PERIOD,
                    );
                let deviated_x: usize = (x as i16 + x_deviation).max(0) as usize;
                let deviated_y: usize = (y as i16 + y_deviation).max(0) as usize;
                let distance_from_coast = deviated_x as f32 - TOTAL_SEA_WIDTH as f32;
                let sea_width_for_river_y =
                    TOTAL_SEA_WIDTH as i32 - sea_deviation_for_river_y as i32;

                let simplex_noise = noise_map[y][x];
                let mut abyssal_plains_depth_adjustment = 0.9;

                if deviated_x < TOTAL_SEA_WIDTH - TRANSITION_PERIOD as usize {
                    // TODO: Ugh this whole section is ugly, could definitely simplify.
                    if deviated_x < ISLANDS_ZONE_WIDTH {
                        let factor = (1.0 * (ISLANDS_ZONE_WIDTH - deviated_x) as f32
                            / TRANSITION_PERIOD as f32)
                            .clamp(0.0, 1.0);
                        abyssal_plains_depth_adjustment += 0.1 * factor;
                        local_max = ISLANDS_MAX_ELEVATION * factor;
                        elevation = pick_value_from_range(
                            simplex_noise,
                            ABYSSAL_PLAINS_MAX_DEPTH * abyssal_plains_depth_adjustment,
                            local_max,
                        );
                    } else {
                        local_max = 0.0;
                        elevation = pick_value_from_range(
                            simplex_noise,
                            ABYSSAL_PLAINS_MAX_DEPTH * abyssal_plains_depth_adjustment,
                            local_max,
                        );
                    }
                } else {
                    let coast_factor =
                        get_boundary_factor(-1.0 * distance_from_coast / TRANSITION_PERIOD as f32);
                    let mut min_inland_elevation = 0.0;
                    let mut max_inland_elevation = 0.0;

                    if deviated_y < NORTH_DESERT_HEIGHT {
                        // TODO: If/else is here because abs() didn't seem to work correctly, need to investigate.
                        let mut factor1: f32 = 0.0;
                        if deviated_x < TOTAL_SEA_WIDTH {
                            factor1 = ((TOTAL_SEA_WIDTH - deviated_x) as f32
                                / TRANSITION_PERIOD as f32)
                                .clamp(0.0, 1.0);
                        } else {
                            factor1 = ((deviated_x - TOTAL_SEA_WIDTH) as f32
                                / TRANSITION_PERIOD as f32)
                                .clamp(0.0, 1.0);
                        }
                        factor1 = factor1.min(deviated_y as f32 / RIVER_Y as f32);
                        factor1 = factor1.min(
                            (NORTH_DESERT_HEIGHT - deviated_y) as f32 / TRANSITION_PERIOD as f32,
                        );
                        factor1 = get_boundary_factor(factor1);

                        let (cx1, cy1) = hex_coordinates_to_cartesian(x as i32, deviated_y as i32);
                        let (cx2, cy2) = hex_coordinates_to_cartesian(
                            TOTAL_SEA_WIDTH as i32 - sea_deviation_for_river_y as i32,
                            RIVER_Y as i32,
                        );
                        // Area is oval-shaped, not circular, with the longer axis running east-west.
                        // I keep fiddling with the ratio of the longer axis to the shorter axis, for awhile I'd settled on 2:1 but
                        // seem to be having trouble making up my mind.
                        let mut factor2 = (cartesian_distance(0.0, cy1, (cx2 - cx1) / 2.0, cy2)
                            / (TRANSITION_PERIOD as f32))
                            .min(1.0);

                        let factor3 = (y as f32 / NORTH_DESERT_HEIGHT as f32).clamp(0.0, 1.0);

                        min_inland_elevation = BOUNDARY_ELEVATION * (1.0 - factor1) * factor2
                            + (1.0 - factor2) * OUTLET_ELEVATION
                            + factor1 * LAKE_MIN_ELEVATION;
                        max_inland_elevation = pick_value_from_range(
                            factor3,
                            FAR_NORTH_DESERT_MAX_ELEVATION,
                            NORTH_DESERT_MAX_ELEVATION,
                        ) * factor2
                            + (1.0 - factor2) * OUTLET_ELEVATION;
                    } else if deviated_y < NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                        let factor1 = ((deviated_y - NORTH_DESERT_HEIGHT) as f32
                            / TRANSITION_PERIOD as f32)
                            .min(1.0);
                        let factor2 = ((NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT - deviated_y)
                            as f32
                            / TRANSITION_PERIOD as f32)
                            .min(1.0);
                        min_inland_elevation = (LAKE_MIN_ELEVATION - BOUNDARY_ELEVATION)
                            * get_boundary_factor(factor1.min(factor2))
                            + BOUNDARY_ELEVATION;
                        max_inland_elevation =
                            (CENTRAL_HIGHLAND_MAX_ELEVATION - NORTH_DESERT_MAX_ELEVATION) * factor1
                                + NORTH_DESERT_MAX_ELEVATION;
                    } else {
                        let factor = ((deviated_y - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT)
                            as f32
                            / TRANSITION_PERIOD as f32)
                            .min(1.0);
                        min_inland_elevation = (LAKE_MIN_ELEVATION - BOUNDARY_ELEVATION)
                            * get_boundary_factor(factor)
                            + BOUNDARY_ELEVATION;
                        max_inland_elevation = (SOUTH_MOUNTAINS_MAX_ELEVATION
                            - CENTRAL_HIGHLAND_MAX_ELEVATION)
                            * factor
                            + CENTRAL_HIGHLAND_MAX_ELEVATION;
                    }

                    let min_elevation =
                        ABYSSAL_PLAINS_MAX_DEPTH * abyssal_plains_depth_adjustment * coast_factor
                            + min_inland_elevation * (1.0 - coast_factor);
                    local_max = max_inland_elevation * (1.0 - coast_factor);

                    elevation = pick_value_from_range(simplex_noise, min_elevation, local_max);

                    // Un-comment this code to see boundaries between regions.
                    // if deviated_y == NORTH_DESERT_HEIGHT || deviated_y == NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT {
                    //     elevation += HEX_SIZE;
                    // }
                }

                // if elevation < 0.0 {
                //     elevation *= get_simplex_noise_for_hex(&simplex, x as f64, (y + HEIGHT_PIXELS * 3) as f64, TOTAL_SEA_WIDTH as f64);
                // }

                // Mostly to prevent very steep coastal cliffs
                // This produces a cut whose downward slope is 2 * MOUNTAINS_MAX_ELEVATION / COAST_WIDTH
                // The cut's deepest point is COAST_WIDTH * cut_factor from the coast, at a depth of 2 * MOUNTAINS_MAX_ELEVATION * cut_factor.
                // Then it slopes upward at the multiplicative inverse of the initial slope (if I did my math right).
                // if distance_from_coast > COAST_WIDTH as f32 * cut_factor {
                //     elevation = elevation.min(distance_from_coast * 2.0 / COAST_WIDTH as f32 * SOUTH_MOUNTAINS_MAX_ELEVATION);
                // } else {
                //     elevation = elevation.min((COAST_WIDTH as f32 * cut_factor - distance_from_coast) * SOUTH_MOUNTAINS_MAX_ELEVATION / COAST_WIDTH as f32 / 2.0 + SOUTH_MOUNTAINS_MAX_ELEVATION * cut_factor * 2.0);
                // }

                // TODO: More realistic seafloor depth, and fix islands.
                // Special features:
                if x == BIG_VOLCANO_X && y == RIVER_Y {
                    // A "volcano" sitting in the river's path. Initially a very tall one-hex column but the angle of repose logic will collapse it,
                    // usually into a cone.
                    // TODO: Make this an event that happens as rainfall is happening, after enough time that I can pick an interesting spot, e.g.
                    // daming the main river, or preventing flow from central area into the north.
                    // Also want more realistic slope, 30-35 degrees.
                    // elevation += get_erruption_elevation(HEX_SIZE * 5.0);
                } else if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE
                    && y <= NE_BASIN_HEIGHT + NE_BASIN_FRINGE
                {
                    // The northeast basin.
                    // TODO: Refactor this so I'm not doing the goofy thing in the rainfall step.
                    elevation = NORTH_DESERT_MAX_ELEVATION * 1.01;
                    // This specifically doesn't include y-deviation so the river source is exactly where we want it to be.
                    if distance_from_river_y != 0 {
                        let factor = ((distance_from_river_y) as f32
                            / (NORTH_DESERT_HEIGHT as f32 - RIVER_Y as f32))
                            .min(1.0);
                        elevation += (NORTH_DESERT_MAX_ELEVATION * 0.01) * factor;
                    }
                }

                if deviated_x <= TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE as usize {
                    elevation +=
                        get_white_noise(seed, x, y) * 0.01 * elevation.max(HEX_SIZE as f32);
                }

                let mut rainfall = 0.0;
                if x < TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                    rainfall = get_rainfall(y, distance_from_coast);
                } else if y < NE_BASIN_HEIGHT {
                    rainfall = NE_BASIN_RAIN;
                }

                if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE {
                    if y <= NE_BASIN_HEIGHT {
                        // if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                        //     rainfall = VERY_HIGH_RAIN;
                        // }
                        if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH + NE_BASIN_FRINGE {
                            // TODO: I don't remember why this is 1.01, probably refactoring would make it clearer.
                            elevation = NORTH_DESERT_MAX_ELEVATION * 1.01;
                        }
                    } else if y <= NE_BASIN_HEIGHT + NE_BASIN_FRINGE {
                        elevation = NORTH_DESERT_MAX_ELEVATION * 1.02;
                    } else if x > TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH {
                        let no_increments: f32 =
                            ((SOUTH_MOUNTAINS_MAX_ELEVATION) / 1000.0).ceil() + 1.0;
                        let boundary = NE_BASIN_HEIGHT + NE_BASIN_FRINGE;
                        let range_size = HEIGHT_PIXELS - boundary;
                        let factor = (1.0
                            - (((y - boundary) as f32 / range_size as f32) * no_increments)
                                .floor()
                                / no_increments)
                            .clamp(0.0, 1.0);
                        elevation =
                            pick_value_from_range(factor, 0.0, SOUTH_MOUNTAINS_MAX_ELEVATION);
                    }
                } /* else {
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
                          let factor = (distance_from_coast as usize + COAST_FRINGE - COAST_WIDTH)
                              as f32
                              / COAST_FRINGE as f32
                              / 2.0;
                          rainfall = get_rainfall(rain_class) * factor
                              + get_rainfall(rain_class + 1) * (1.0 - factor);
                      }
                  } */

                // As above, ignore y-deviation.
                if distance_from_river_y == 0 {
                    elevation = elevation.min(NORTH_DESERT_MAX_ELEVATION * 1.01);
                }

                if elevation > 0.0 {
                    // TODO: Two things we want to compensate for here: sea level rise and erosion due to rainfall.
                    // Going to test just doing the first thing, see how much I need for second thing.
                    // uplift = MAX_UPLIFT * elevation / SOUTH_MOUNTAINS_MAX_ELEVATION;
                    uplift = (0.02 * YEARS_PER_STEP + RAIN_BASED_UPLIFT_FACTOR * rainfall) * elevation / local_max;
                }

                if uplift.is_nan() {
                    println!("Uplift is NaN: {}", uplift);
                    println!("elevation: {}", elevation);
                    println!("x: {}, y: {}", x, y);
                    panic!("Uplift is NaN");
                }

                hex_map[y as usize].push(Hex {
                    coordinate: (x, y),
                    elevation,
                    water_depth: 0.0,
                    suspended_load: 0.0,
                    rainfall: rainfall * RAINFALL_FACTOR,
                    // TODO: Fiddle with range, seems to help with coastlines and mountains but may make chanelization worse.
                    erosion_multiplier: 0.90
                        + get_simplex_noise_for_hex(
                            &simplex,
                            x as f64,
                            (y + HEIGHT_PIXELS * 2) as f64,
                            1.0,
                        ) * 0.05
                        + get_simplex_noise_for_hex(
                            &simplex,
                            (x + WIDTH_HEXAGONS) as f64,
                            (y + HEIGHT_PIXELS * 2) as f64,
                            3.0,
                        ) * 0.05
                        + get_simplex_noise_for_hex(
                            &simplex,
                            (x + WIDTH_HEXAGONS * 2) as f64,
                            (y + HEIGHT_PIXELS * 2) as f64,
                            7.0,
                        ) * 0.05
                        + get_simplex_noise_for_hex(
                            &simplex,
                            (x + WIDTH_HEXAGONS * 3) as f64,
                            (y + HEIGHT_PIXELS * 2) as f64,
                            29.0,
                        ) * 0.05,
                    uplift,
                    original_land: elevation > BASE_SEA_LEVEL,
                });
            }
        }

        let_slopes_settle(&mut hex_map);
        fill_sea(&mut hex_map);

        // Calculate initial max and average elevation for comparison during simulation
        // Using the continental hex definition for consistency with logging
        let continental = calculate_continental_hexes(&hex_map, BASE_SEA_LEVEL);

        // Regional boundaries based on y coordinate
        let central_boundary = NORTH_DESERT_HEIGHT;
        let south_boundary = NORTH_DESERT_HEIGHT + CENTRAL_HIGHLAND_HEIGHT;

        let (
            initial_max_elevation,
            total_land,
            continental_count,
            north_total,
            north_count,
            central_total,
            central_count_region,
            south_total,
            south_count,
        ): (f32, f64, usize, f64, usize, f64, usize, f64, usize) = {
            let mut max_elev = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            let mut count = 0usize;
            let mut n_sum = 0.0f64;
            let mut n_count = 0usize;
            let mut c_sum = 0.0f64;
            let mut c_count = 0usize;
            let mut s_sum = 0.0f64;
            let mut s_count = 0usize;

            for y in 0..HEIGHT_PIXELS {
                for x in 0..WIDTH_HEXAGONS {
                    if continental[y][x] {
                        let elev = hex_map[y][x].elevation;
                        let elev_above_sea = (elev - BASE_SEA_LEVEL) as f64;

                        if elev > max_elev {
                            max_elev = elev;
                        }
                        sum += elev_above_sea;
                        count += 1;

                        // Regional breakdown
                        if y < central_boundary {
                            n_sum += elev_above_sea;
                            n_count += 1;
                        } else if y < south_boundary {
                            c_sum += elev_above_sea;
                            c_count += 1;
                        } else {
                            s_sum += elev_above_sea;
                            s_count += 1;
                        }
                    }
                }
            }
            // Subtract sea level from max to get elevation above sea level
            (
                max_elev - BASE_SEA_LEVEL,
                sum,
                count,
                n_sum,
                n_count,
                c_sum,
                c_count,
                s_sum,
                s_count,
            )
        };

        let initial_avg_elevation = if continental_count > 0 {
            (total_land / continental_count as f64) as f32
        } else {
            0.0
        };

        let initial_north_avg = if north_count > 0 {
            (north_total / north_count as f64) as f32
        } else {
            0.0
        };
        let initial_central_avg = if central_count_region > 0 {
            (central_total / central_count_region as f64) as f32
        } else {
            0.0
        };
        let initial_south_avg = if south_count > 0 {
            (south_total / south_count as f64) as f32
        } else {
            0.0
        };

        // Calculate initial sea hex average elevation
        // Sea hexes: x <= TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH, not continental, elevation < sea level
        let (sea_total, sea_count): (f64, usize) = {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            let sea_boundary_x = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH;

            for y in 0..HEIGHT_PIXELS {
                for x in 0..=sea_boundary_x {
                    if !continental[y][x] && hex_map[y][x].elevation < BASE_SEA_LEVEL {
                        sum += hex_map[y][x].elevation as f64;
                        count += 1;
                    }
                }
            }
            (sum, count)
        };

        let initial_sea_avg_elevation = if sea_count > 0 {
            (sea_total / sea_count as f64) as f32
        } else {
            0.0
        };

        let hex_duration = hex_start.elapsed();
        println!("Hex map creation took: {:?}", hex_duration);
        println!(
            "Initial max elevation: {:.3} ft, initial avg elevation: {:.3} ft, initial sea avg elevation: {:.3} ft",
            initial_max_elevation, initial_avg_elevation, initial_sea_avg_elevation
        );
        println!(
            "  regional initial avgs: north {:.3}, central {:.3}, south {:.3}",
            initial_north_avg, initial_central_avg, initial_south_avg
        );

        // Generate initial terrain and rainfall PNGs
        let mut init_frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];
        render_frame(&hex_map, &mut init_frame_buffer, BASE_SEA_LEVEL, false);
        save_buffer_png(
            "terrain_initial.png",
            &init_frame_buffer,
            WIDTH_PIXELS as u32,
            HEIGHT_PIXELS as u32,
        );
        println!("Saved terrain_initial.png");

        // Rainfall map - max expected is ~95.8 * RAINFALL_FACTOR
        // let max_expected_rainfall = 95.8 * RAINFALL_FACTOR;
        // render_rainfall(&hex_map, &mut init_frame_buffer, max_expected_rainfall);
        // save_buffer_png(
        //     "rainfall.png",
        //     &init_frame_buffer,
        //     WIDTH_PIXELS as u32,
        //     HEIGHT_PIXELS as u32,
        // );
        // println!("Saved rainfall.png");

        (
            hex_map,
            seed,
            0,
            initial_max_elevation,
            initial_avg_elevation,
            initial_sea_avg_elevation,
            initial_north_avg,
            initial_central_avg,
            initial_south_avg,
            0.0,
        )
    };

    println!("Seed: {}", seed);
    println!("Starting step: {}", starting_step);

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    let total_steps = (WIDTH_HEXAGONS as u32) * rounds;
    let remaining_steps = total_steps.saturating_sub(starting_step);
    let (final_sea_level, final_elapsed_secs) = simulate_erosion(
        &mut hex_map,
        remaining_steps,
        seed,
        starting_step,
        initial_max_elevation,
        initial_avg_elevation,
        initial_sea_avg_elevation,
        initial_north_avg,
        initial_central_avg,
        initial_south_avg,
        prior_elapsed_secs,
    );

    // TODO: This isn't working, should fix.
    // Count final blue pixels for quick sanity check
    let final_blue = frame_buffer
        .iter()
        .filter(|&&px| {
            (px & 0x0000FF) == 0x0000FF && (px >> 16 & 0xFF) == 0 && (px >> 8 & 0xFF) == 0
        })
        .count();
    println!("Final blue pixels: {}", final_blue);

    // Time PNG conversion
    let png_start = Instant::now();
    render_frame(&mut hex_map, &mut frame_buffer, final_sea_level, true);
    save_buffer_png(
        "terrain_water.png",
        &frame_buffer,
        WIDTH_PIXELS as u32,
        HEIGHT_PIXELS as u32,
    );
    let final_step = starting_step + remaining_steps;
    save_simulation_state_csv(
        "terrain_water_final.csv",
        &hex_map,
        seed,
        final_step,
        initial_max_elevation,
        initial_avg_elevation,
        initial_sea_avg_elevation,
        initial_north_avg,
        initial_central_avg,
        initial_south_avg,
        final_elapsed_secs,
    );

    render_frame(&mut hex_map, &mut frame_buffer, final_sea_level, false);
    save_buffer_png(
        "terrain.png",
        &frame_buffer,
        WIDTH_PIXELS as u32,
        HEIGHT_PIXELS as u32,
    );
    save_simulation_state_csv(
        "terrain_final.csv",
        &hex_map,
        seed,
        final_step,
        initial_max_elevation,
        initial_avg_elevation,
        initial_sea_avg_elevation,
        initial_north_avg,
        initial_central_avg,
        initial_south_avg,
        final_elapsed_secs,
    );

    let save_duration = png_start.elapsed();
    println!("Image rendering and saving took: {:?}", save_duration);
    println!("Terrain visualization saved as terrain.png");
}
