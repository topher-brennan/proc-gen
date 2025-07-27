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
    coordinate: (u16, u16),
    elevation: f32, // Feet
    water_depth: f32, // Feet of water currently stored in this hex
    suspended_load: f32, // Feet of sediment stored in water column
}

fn elevation_to_color(elevation: f32) -> Rgb<u8> {
    if elevation < SEA_LEVEL {
        let normalized_elevation = (elevation / SEA_LEVEL).min(1.0);
        if normalized_elevation < 0.0 {
            // Black, to mark where errosion has lowered elevation below the lowest possible initial elevation.
            Rgb([0, 0, 0])
        } else {
            // Purple to light blue
            let red = 128;
            let green = (255.0 * normalized_elevation) as u8;
            let blue = (128.0 + (127.0 * normalized_elevation)) as u8;
            Rgb([red, green, blue])
        }
    } else {
        // Land: green -> yellow -> orange -> red -> brown -> white
        let land_height = elevation - SEA_LEVEL;
        let max_land_height = MAX_ELEVATION - SEA_LEVEL;
        let normalized_height = (land_height / max_land_height).min(1.0);
        
        if normalized_height < 0.225 {
            // Green to yellow
            let factor = normalized_height / 0.225;
            let red = (255.0 * factor) as u8;
            let green = 255;
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.45 {
            // Yellow (255,255,0) → Orange (255,127,0)
            let factor = (normalized_height - 0.225) / 0.225; // 0..1
            let red   = 255;
            let green = (127.0 + 128.0 * (1.0 - factor)) as u8; // 255→127
            let blue  = 0;
            Rgb([red, green, blue])
        // TODO: The later stages of this transition don't subjectively look right to me,
        // is there some standard way to do this?
        } else if normalized_height < 0.675 {
            // Orange (255,127,0) to Red (255,0,0)
            let factor = (normalized_height - 0.45) / 0.225; // 0..1
            let red = 255;
            let green = (127.0 * (1.0 - factor)) as u8; // 127→0
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.9 {
            // Red to brown
            let factor = (normalized_height - 0.675) / 0.225;
            let red = 62 + (193.0 * (1.0 - factor)) as u8; // 255→62
            let green = 28 * factor as u8;
            let blue = 0;
            Rgb([red, green, blue])
        } else {
            // Brown to white
            let factor = (normalized_height - 0.9) / 0.1;
            let red = 62 + (193.0 * factor) as u8; // 62→255
            let green = 28 + (237.0 * factor) as u8;
            let blue = (255.0 * factor) as u8;
            // TODO: I seem to be getting some weird magenta dots in the output, not sure why.
            Rgb([red, green, blue])
        }
    }
}

// TODO: Vary rainfall so that there's more rainfall further south, and more rainfall closer to the coast.
// TODO: Visualize results.
// ------------------------------------------------------------
// Simple rainfall–runoff experiment (Milestone 1)
// ------------------------------------------------------------
fn simulate_rainfall(
    hex_map: &mut Vec<Vec<Hex>>,
    steps: u32,
    river_y: usize,
) {
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

    let mut gpu_data: Vec<HexGpu> = Vec::with_capacity(width * height);
    for row in hex_map.iter() {
        for h in row {
            gpu_data.push(HexGpu {
                elevation: h.elevation,
                water_depth: h.water_depth,
                suspended_load: h.suspended_load,
                _padding: 0.0,
            });
        }
    }

    gpu_sim.upload_data(&gpu_data);

    for _step in 0..steps {
        // Mass balance stats per step
        let rainfall_added = (width * height) as f32 * RAIN_PER_STEP;
        let mut step_outflow = 0.0f32;
        let mut step_sediment_in = 0.0f32;
        let mut step_sediment_out = 0.0f32;

        // 1) Add rainfall uniformly – GPU implementation
        {
            gpu_sim.run_rainfall_step(RAIN_PER_STEP, width * height);
            let updated = gpu_sim.download_data();

            // Write back updated water depths
            for (idx, ghex) in updated.iter().enumerate() {
                let y = idx / width;
                let x = idx % width;
                hex_map[y][x].water_depth = ghex.water_depth;
            }
        }

        // GPU water routing --------------------------------------------------
        gpu_sim.run_water_routing_step(width, height, FLOW_FACTOR, MAX_FLOW);
        gpu_sim.run_scatter_step(width, height);
        // --- GPU min-slope + erosion/deposition passes ---
        gpu_sim.run_min_neigh_step(width, height);
        gpu_sim.run_erosion_step(width, height);

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

        // Apply ocean boundary condition on the west edge (x == 0)
        for y in 0..height {
            let cell = &mut hex_map[y][0];
            let target_depth = (SEA_LEVEL - cell.elevation).max(0.0);
            if cell.water_depth > target_depth {
                let surplus = cell.water_depth - target_depth;
                total_outflow += surplus;
                step_outflow += surplus;
            }
            total_sediment_out += cell.suspended_load;
            step_sediment_out += cell.suspended_load;

            cell.water_depth = target_depth;
            cell.suspended_load = 0.0; // Flushed to sea

            // --- FIX: Update the GPU buffer to reflect the outflow ---
            let cell_idx = y * width; // x is always 0
            gpu_sim.add_inflow(cell_idx, cell.water_depth, cell.suspended_load);
        }

        // 1b) Add river inflow at east edge AFTER GPU routing scatter so it isn't overwritten
        if river_y < height {
            apply_river_inflow(river_y, hex_map, &mut gpu_sim);
            let suspended_load_per_step = RIVER_WATER_PER_STEP * RIVER_LOAD_FACTOR * KC / HEX_SIZE;
            step_sediment_in += suspended_load_per_step;
            total_sediment_in += suspended_load_per_step;
        }

        if _step % (WIDTH_HEXAGONS as u32) == (WIDTH_HEXAGONS as u32) - 1 {
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

            let mean_depth = water_on_land / cells_above_sea_level as f32;

            let wet_cells: usize = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.elevation > SEA_LEVEL && h.water_depth > WATER_THRESHOLD).count())
                .sum();

            let wet_cells_percentage = wet_cells as f32 / cells_above_sea_level as f32 * 100.0;
            let round = _step / (WIDTH_HEXAGONS as u32);

            // render_frame(hex_map, frame_buffer, river_y);
            // save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
            // save_png("terrain.png", hex_map);

            println!(
                "Round {:.0}: water in {:.1}  water out {:.1}  stored {:.0}  mean depth {:.2} ft  max depth {:.2} ft  wet {:} ({:.1}%)  sediment in {:.3}  sediment out {:.3}",
                round,
                (rainfall_added + RIVER_WATER_PER_STEP),
                step_outflow,
                water_on_land,
                mean_depth,
                max_depth,
                wet_cells,
                wet_cells_percentage,
                step_sediment_in,
                step_sediment_out,
            );
        }
    }

    let water_remaining: f32 = hex_map
        .iter()
        .flat_map(|row| row.iter())
        .map(|h| h.water_depth)
        .sum();

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft-hexes, water remaining on land: {:.2} ft-hexes, sediment in {:.1},  sediment out {:.1}",
        steps, total_outflow, water_remaining, total_sediment_in, total_sediment_out
    );
}

// --- Helper: apply river inflow at east edge ---
fn apply_river_inflow(river_y: usize,
                       hex_map: &mut [Vec<Hex>],
                       gpu_sim: &mut GpuSimulation
) {
    if river_y >= hex_map.len() { return; }
    let rx = WIDTH_HEXAGONS as usize - 1;
    hex_map[river_y][rx].water_depth += RIVER_WATER_PER_STEP;

    let suspended_load_per_step = RIVER_WATER_PER_STEP * RIVER_LOAD_FACTOR * KC / HEX_SIZE;
    hex_map[river_y][rx].suspended_load += suspended_load_per_step;

    let idx = river_y*(WIDTH_HEXAGONS as usize) + (WIDTH_HEXAGONS as usize) - 1;
    gpu_sim.add_water(idx, RIVER_WATER_PER_STEP);
    gpu_sim.add_load(idx, suspended_load_per_step);
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
    let _ = img.save(path);
}

fn save_png(path: &str, hex_map: &Vec<Vec<Hex>>) {
    // Create the visualization image
    let mut img = ImageBuffer::new(WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
    
    // For each pixel, find the nearest hex and use its elevation
    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            // Convert pixel coordinates to hex coordinates
            // This is a simple mapping - you might want to adjust this based on your hex layout
            let hex_x = (x as f32 / HEX_FACTOR) as u16;
            let hex_y = y;
            
            // Clamp to valid hex coordinates
            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);
            
            // Get the elevation from the hex map
            let elevation = hex_map[hex_y as usize][hex_x as usize].elevation;
            
            // Convert elevation to color
            let color = elevation_to_color(elevation);
            
            // Set the pixel
            img.put_pixel(x as u32, y as u32, color);
        }
    }

    img.save(path).expect("Failed to save image");
}

// Renders current hex_map state into an RGB buffer (u32 per pixel)
fn render_frame(hex_map: &Vec<Vec<Hex>>, buffer: &mut [u32], _river_y: usize) {
    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            let hex_x = ((x as f32) / HEX_FACTOR) as u16;
            let hex_y = y;

            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);

            let hex = &hex_map[hex_y as usize][hex_x as usize];
            // Choose colour – highlight water depth strongly so it stands out
            let color = if hex.water_depth > WATER_THRESHOLD {
                // Strong blue for water for debugging
                let blue = 255u8;
                let g = 0u8;
                let r = 0u8;
                (r as u32) << 16 | (g as u32) << 8 | (blue as u32)
            } else {
                let Rgb([r, g, b]) = elevation_to_color(hex.elevation);
                let r = (r as f32 * 0.4) as u8;
                let g = (g as f32 * 0.4) as u8;
                let b = (b as f32 * 0.4) as u8;
                (r as u32) << 16 | (g as u32) << 8 | (b as u32)
            };
            buffer[(y as usize) * (WIDTH_PIXELS as usize) + (x as usize)] = color;
        }
    }
}

fn main() {
    // Allow user to override number of rounds via command-line: first positional arg is rounds, e.g. `cargo run --release -- 2000`
    let rounds: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_ROUNDS);

    let mut hex_map = Vec::new();
    let mut rng = rand::thread_rng();

    // Time hex map creation
    let hex_start = Instant::now();
    let river_y = (0.29 * HEIGHT_PIXELS as f32) as usize;
    for y in 0..HEIGHT_PIXELS {
        hex_map.push(Vec::new());
        for x in 0..WIDTH_HEXAGONS {
            let mut southern_multiplier = 1.0;
            let southern_threshold = (river_y + 276) as u16;
            if y > southern_threshold {
                southern_multiplier += 0.5 * ((y - southern_threshold) as f32 / 1518.0).max(0.0);
            }

            let x_based_elevation = (x as f32) * 4.0 * southern_multiplier;

            let mut elevation = x_based_elevation;
            if y == 0 || y == HEIGHT_PIXELS - 1 || x == WIDTH_HEXAGONS - 1 {
                // Try to prevent weird artifacts at the edges of the map. Not applied to western edge because
                // we have the outflow mechanic there.
                elevation += rng.gen_range(HEX_SIZE/200.0..HEX_SIZE/100.0);
            } else {
                elevation += rng.gen_range(0.0..HEX_SIZE/100.0);
            }

            ensure_finite!(elevation, "elevation", x, y, -1);

            let mut water_depth = 0.0;
            // This allows for the possibility of pockets of dry land below sea level. It errs on the side of
            // starting with zero water, I could probably do something fancier with pathfinding to guarantee
            // hexes start with water IFF they have a path to the sea.
            if x_based_elevation + HEX_SIZE / 100.0 < SEA_LEVEL {
                water_depth = SEA_LEVEL - elevation;
            }
            hex_map[y as usize].push(Hex {
                coordinate: (x, y),
                elevation,
                water_depth,
                suspended_load: 0.0,
            });
        }
    }

    // ---------------------------------------------------------------------
    // Prefill local pits (cells lower than all neighbours but still above
    // sea level) with enough water so the initial water surface equals the
    // lowest neighbour.  This prevents long "filling" spin-up times for tiny
    // closed depressions.
    // ---------------------------------------------------------------------
    for y in 0..HEIGHT_PIXELS as usize {
        for x in 0..WIDTH_HEXAGONS as usize {
            let cell_elev = hex_map[y][x].elevation;
            if cell_elev < SEA_LEVEL { continue; } // below sea already filled

            let offsets = if (x & 1) == 0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
            let mut lowest_neigh = f32::INFINITY;
            let mut is_pit = true;

            for &(dx, dy) in offsets {
                let nx = x as i32 + dx as i32;
                let ny = y as i32 + dy as i32;
                if nx < 0 || nx >= WIDTH_HEXAGONS as i32 || ny < 0 || ny >= HEIGHT_PIXELS as i32 { continue; }
                let n_elev = hex_map[ny as usize][nx as usize].elevation;
                if n_elev <= cell_elev {
                    is_pit = false;
                    break;
                }
                if n_elev < lowest_neigh { lowest_neigh = n_elev; }
            }

            if is_pit && lowest_neigh.is_finite() {
                hex_map[y][x].water_depth = lowest_neigh - cell_elev;
            }
        }
    }

    // --------------------------------------------------------
    // Milestone 1: pure rainfall with fixed sea boundary
    // --------------------------------------------------------

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    // Maybe KC should be around 0.00574 to harmonize these?
    let total_steps = (WIDTH_HEXAGONS as u32) * rounds;
    simulate_rainfall(&mut hex_map, total_steps, river_y);

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
    render_frame(&mut hex_map, &mut frame_buffer, river_y);
    save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
    save_png("terrain.png", &hex_map);

    let save_duration = png_start.elapsed();
    println!("Image rendering and saving took: {:?}", save_duration);
    println!("Terrain visualization saved as terrain.png");
}

#[cfg(test)]
mod river_tests {
    use super::*;
    #[test]
    fn river_inflow_correct() {
        let width = WIDTH_HEXAGONS as usize;
        let height = 3usize;
        let river_y = 1usize;
        let mut hex_map: Vec<Vec<Hex>> = (0..height).map(|y| {
            (0..width).map(|x| Hex{coordinate:(x as u16,y as u16), elevation:0.0, water_depth:0.0, suspended_load:0.0}).collect()
        }).collect();
        let mut next_water: Vec<Vec<f32>> = (0..height).map(|_| vec![0.0; width]).collect();
        let mut next_load: Vec<Vec<f32>>  = (0..height).map(|_| vec![0.0; width]).collect();

        apply_river_inflow(river_y, &mut hex_map, &mut next_water, &mut next_load, &mut gpu_sim);
        let rx = width-1;
        assert!((hex_map[river_y][rx].water_depth - RIVER_WATER_PER_STEP).abs() < 1e-6);
        assert!((next_water[river_y][rx] - RIVER_WATER_PER_STEP).abs() < 1e-6);
        let expected_load = RIVER_WATER_PER_STEP * RIVER_LOAD_FACTOR * KC / HEX_SIZE;
        assert!((hex_map[river_y][rx].suspended_load - expected_load).abs() < 1e-6);
        assert!((next_load[river_y][rx] - expected_load).abs() < 1e-6);
        // ensure neighbouring cell unchanged
        assert_eq!(hex_map[river_y][rx-1].water_depth, 0.0);
    }
}