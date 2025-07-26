use rand::Rng;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use rayon::prelude::*;
use image::{RgbImage};
mod gpu_simulation;
use gpu_simulation::{GpuSimulation, HexGpu};
use pollster;

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

// Sentinel indicating no downslope target during water-routing
const NO_TARGET: usize = usize::MAX;

// const HEIGHT_PIXELS: u16 = 2160;
// const WIDTH_PIXELS: u16 = 3840;
const HEIGHT_PIXELS: u16 = 216;
const WIDTH_PIXELS: u16 = 384;

// Approximate value of sqrt(3) / 2
// Useful because if the length of a perpendicular line segment connecting two
// sides of a regular hexagon is 1, then the length of a line segment
// connecting the corners of the hexagon is 2 / sqrt(3), and the side length is
// 1 / sqrt(3). These values are equal to 2 * sqrt(3) / 3 and sqrt(3) / 3,
// respectively, so their average is ([2 + 1] * sqrt(3)) / 3 / 2 = sqrt(3) / 2
// This is the area and average width of the hexagon.
const HEX_FACTOR: f32 = 0.8660254037844386;

// Assume 1 pixel per hex vertically, and HEX_FACTOR pixels per hex horizontally.
// const WIDTH_HEXAGONS: u16 = 4434;
const WIDTH_HEXAGONS: u16 = (WIDTH_PIXELS as f32 / HEX_FACTOR) as u16;

// This combined wtih some other things keeps the starting coastline in the middle of the map.
const SEA_LEVEL: f32 = (WIDTH_HEXAGONS as f32) * 2.0;
// This can be used to convert volume from foot-hexes to cubic feet.
const HEX_SIZE: f32 = 2640.0; // Feet
// One hour worth of average discharge from the Aswan Dam.
const RIVER_WATER_PER_STEP: f32 = 37.1; // Feet
// I *think* this should be the elevation-drop-per-hex that gives us a steady state,
// but now I'm worried I did my math wrong.
const RIVER_LOAD_FACTOR: f32 = 1.2;
// One inch of rain per year - desert conditions.
const RAIN_PER_STEP: f32 = 1.0 / 12.0 / 365.0 / 24.0; // Feet
const DEFAULT_ROUNDS: u32 = 1000; // default number of "rounds" (each = WIDTH_HEXAGONS steps)
const WATER_THRESHOLD: f32 = 1.0 / 12.0; // One inch in feet

// --- Minimal erosion / deposition constants ---
// Questioning the decision to divide by HEX_SIZE in errosion calculations.
const KC: f32 = HEX_SIZE / 100.0; // capacity coefficient
const KE: f32 = 0.05;  // erosion rate fraction
const KD: f32 = 0.05;  // deposition rate fraction
// How large FLOW_FACTOR can be without erroding the sea floor seems related to
// KE and KD, but I'm not sure of the exact relationship.
const FLOW_FACTOR: f32 = 0.9;
const MAX_SLOPE: f32 = 1.0; // Prevents runaway erosion by capping slope used in capacity calc
const MAX_FLOW: f32 = WIDTH_HEXAGONS as f32;
const MAX_ELEVATION: f32 = (WIDTH_HEXAGONS as f32) * 4.0 + HEX_SIZE / 100.0;

struct Hex {
    coordinate: (u16, u16),
    elevation: f32, // Feet
    water_depth: f32, // Feet of water currently stored in this hex
    suspended_load: f32, // Feet of sediment stored in water column
}

// Returns the 6 neighbors of a hexagon, assuming a "columns line up" layout,
// such that (0, 1) is at the "6 o'clock" of (0, 0), and (1, 0) is at the
// "4 o'clock" of (0, 0).
fn hex_neighbors(coordinate: (u16, u16)) -> Vec<(u16, u16)> {
    let mut neighbors = Vec::new();
    neighbors.push((coordinate.0.wrapping_add(1), coordinate.1));
    neighbors.push((coordinate.0, coordinate.1.wrapping_add(1)));
    neighbors.push((coordinate.0.wrapping_sub(1), coordinate.1));
    neighbors.push((coordinate.0, coordinate.1.wrapping_sub(1)));

    if (coordinate.0 % 2) == 0 {
        // (x-1, y) and (x+1, y) represent the "4 o'clock" and "8 o'clock", so
        // we need the "2 o'clock" and "10 o'clock" neighbors.
        neighbors.push((coordinate.0.wrapping_sub(1), coordinate.1.wrapping_sub(1)));
        neighbors.push((coordinate.0.wrapping_add(1), coordinate.1.wrapping_sub(1)));
    } else {
        // (x-1, y) and (x+1, y) represent the "2 o'clock" and "10 o'clock", so
        // we need the "4 o'clock" and "8 o'clock" neighbors.
        neighbors.push((coordinate.0.wrapping_sub(1), coordinate.1.wrapping_add(1)));
        neighbors.push((coordinate.0.wrapping_add(1), coordinate.1.wrapping_add(1)));
    }

    filter_coordinates(neighbors)
}

// TODO: This might be me being used to languages with more syntactic sugar,
// but I wonder if there's a more concise/idiomatic way to do this.
// Make sure x >= 0, x < WIDTH_HEXAGONS, y >= 0, y < HEIGHT_PIXELS
fn filter_coordinates(coordinates: Vec<(u16, u16)>) -> Vec<(u16, u16)> {
    let mut filtered = Vec::new();
    for (x, y) in coordinates {
        if x < WIDTH_HEXAGONS && y < HEIGHT_PIXELS {
            filtered.push((x, y));
        }
    }
    filtered
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
    frame_buffer: &mut Vec<u32>,
) {
    let height = hex_map.len();
    if height == 0 {
        return;
    }
    let width = hex_map[0].len();

    // ---------------------------------------------------------
    // GPU helper initialisation (only used for rainfall phase)
    // ---------------------------------------------------------
    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    gpu_sim.initialize_buffer(width, height);

    let mut total_outflow = 0.0f32;
    let mut total_sediment_in = 0.0f32;
    let mut total_sediment_out = 0.0f32;

    // Reusable buffer for next water depths
    let mut next_water: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    // Reusable buffer for sediment transport (same layout)
    let mut next_load: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    // Buffer storing minimum neighbour elevation for each cell
    let mut min_neigh_elev: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    // Reusable buffer for elevation deltas (angle-of-repose)
    let mut delta_elev: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    // --- Buffers for parallel water-routing (gather→scatter) ---
    #[derive(Clone)]
    struct RowOut {
        w:    Vec<f32>,
        load: Vec<f32>,
        tgt:  Vec<usize>,
    }

    let mut out_rows: Vec<RowOut> = (0..height)
        .map(|_| RowOut {
            w:    vec![0.0f32; width],
            load: vec![0.0f32; width],
            tgt:  vec![NO_TARGET; width],
        })
        .collect();

    for _step in 0..steps {
        // Mass balance stats per step
        let rainfall_added = (width * height) as f32 * RAIN_PER_STEP;
        let mut step_outflow = 0.0f32;
        let mut step_sediment_in = 0.0f32;
        let mut step_sediment_out = 0.0f32;

        // 0) Pre-compute min neighbour elevation in parallel (for slope calc)
        min_neigh_elev
            .par_iter_mut()
            .enumerate()
            .for_each(|(y, row)| {
                let y_i = y as i32;
                let width_i = width as i32;
                let height_i = height as i32;
                let src_row = &hex_map[y];
                for x in 0..width {
                    let cell_elev = src_row[x].elevation;
                    let offsets = if (x & 1) == 0 {
                        &NEIGH_OFFSETS_EVEN
                    } else {
                        &NEIGH_OFFSETS_ODD
                    };
                    let mut min_n = cell_elev;
                    for &(dx, dy) in offsets {
                        let nx = x as i32 + dx as i32;
                        let ny = y_i + dy as i32;
                        if nx < 0 || nx >= width_i || ny < 0 || ny >= height_i {
                            continue;
                        }
                        let n_elev = hex_map[ny as usize][nx as usize].elevation;
                        if n_elev < min_n {
                            min_n = n_elev;
                        }
                    }
                    row[x] = min_n;
                }
            });

        // 1) Add rainfall uniformly – GPU implementation
        {
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
            gpu_sim.run_rainfall_step(RAIN_PER_STEP, width * height);
            let updated = gpu_sim.download_data();

            // Write back updated water depths
            for (idx, ghex) in updated.iter().enumerate() {
                let y = idx / width;
                let x = idx % width;
                hex_map[y][x].water_depth = ghex.water_depth;
            }
        }

        // 1b) Add river inflow at east edge (x = WIDTH_HEXAGONS-1)
        if river_y < height {
            hex_map[river_y][WIDTH_HEXAGONS as usize - 1].water_depth += RIVER_WATER_PER_STEP;

            let suspended_load_per_step = RIVER_WATER_PER_STEP * RIVER_LOAD_FACTOR * KC / HEX_SIZE;
            hex_map[river_y][WIDTH_HEXAGONS as usize - 1].suspended_load += suspended_load_per_step;
            step_sediment_in += suspended_load_per_step;
            total_sediment_in += suspended_load_per_step;
        }

        // 2) Zero reusable buffers in parallel
        next_water.par_iter_mut().for_each(|row| row.fill(0.0));
        next_load.par_iter_mut().for_each(|row| row.fill(0.0));
        out_rows.par_iter_mut().for_each(|row| {
            row.w.fill(0.0);
            row.load.fill(0.0);
            row.tgt.fill(NO_TARGET);
        });

        // 3a) Phase-1: gather outflow per cell (parallel, read-only on hex_map)
        out_rows.par_iter_mut().enumerate().for_each(|(y, row)| {
            for x in 0..width {
                let cell = &hex_map[y][x];
                let w = cell.water_depth;
                if w <= 0.0 {
                    continue;
                }

                let mut min_height = cell.elevation + w;
                let mut tgt: Option<(usize, usize)> = None;
                let offsets = if (x & 1) == 0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
                for &(dx, dy) in offsets {
                    let nx = x as i32 + dx as i32;
                    let ny = y as i32 + dy as i32;
                    if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 { continue; }
                    let n_hex = &hex_map[ny as usize][nx as usize];
                    let nh = n_hex.elevation + n_hex.water_depth;
                    if nh < min_height {
                        min_height = nh;
                        tgt = Some((nx as usize, ny as usize));
                    }
                }

                if let Some((tx, ty)) = tgt {
                    let target_hex = &hex_map[ty][tx];
                    let diff = cell.elevation + w - (target_hex.elevation + target_hex.water_depth);
                    let move_w = (if diff > w { w } else { diff * FLOW_FACTOR }).min(MAX_FLOW);
                    if move_w > 0.0 {
                        row.w[x] = move_w;
                        row.load[x] = cell.suspended_load * move_w / w;
                        row.tgt[x] = ty * width + tx;
                    }
                }
            }
        });

        // 3b) Phase-2: scatter – assemble inflows & own remainder (parallel)
        next_water
            .par_iter_mut()
            .zip(&mut next_load)
            .enumerate()
            .for_each(|(y, (row_next_w, row_next_load))| {
                for x in 0..width {
                    let mut new_w   = hex_map[y][x].water_depth   - out_rows[y].w[x];

                    let suspended_load = hex_map[y][x].suspended_load;
                    let out_load = out_rows[y].load[x];
                    let mut new_load = suspended_load - out_load;

                    // Clamp to avoid runaway depths or negative values
                    new_w   = new_w.max(0.0);
                    new_load= new_load.max(0.0);

                    ensure_finite!(new_w, "new_w", x, y, _step);
                    ensure_finite!(new_load, format!("new_load ({suspended_load} - {out_load})"), x, y, _step);

                    let cur_idx = y * width + x;
                    let offsets = if (x & 1) == 0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
                    for &(dx, dy) in offsets {
                        let nx = x as i32 + dx as i32;
                        let ny = y as i32 + dy as i32;
                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 { continue; }
                        let nxi = nx as usize;
                        let nyi = ny as usize;
                        if out_rows[nyi].tgt[nxi] == cur_idx {
                            new_w   += out_rows[nyi].w   [nxi];
                            new_load+= out_rows[nyi].load[nxi];
                        }
                    }

                    row_next_w[x]    = new_w;
                    row_next_load[x] = new_load;
                }
            });

        // 4) Apply next water depths, counting outflow at sea boundary (x == 0)
        for y in 0..height {
            for x in 0..width {
                let new_w = next_water[y][x];
                let new_load = next_load[y][x];
                if x == 0 {
                    // West edge: ocean boundary keeps water_surface = SEA_LEVEL
                    let target_depth = (SEA_LEVEL - hex_map[y][x].elevation).max(0.0);
                    // Any water above that level leaves the domain
                    if new_w > target_depth {
                        let surplus = new_w - target_depth;
                        total_outflow += surplus;
                        step_outflow += surplus;
                    }
                    // Set water depth to ocean equilibrium
                    hex_map[y][x].water_depth = target_depth;

                    total_sediment_out += hex_map[y][x].suspended_load;
                    step_sediment_out += hex_map[y][x].suspended_load;
                    hex_map[y][x].suspended_load = 0.0; // flushed to sea
                } else {
                    // First gather immutable info
                    let cell_elev = hex_map[y][x].elevation;
                    let min_n = min_neigh_elev[y][x];

                    // Now mutate cell safely
                    let cell = &mut hex_map[y][x];
                    cell.water_depth = new_w;
                    cell.suspended_load = new_load;

                    let slope = ((cell_elev - min_n) / HEX_SIZE).max(0.0);
                    ensure_finite!(slope, format!("slope (({cell_elev} - {min_n}) / {HEX_SIZE})"), x, y, _step);

                    let capacity = KC * cell.water_depth * slope.min(MAX_SLOPE);
                    ensure_finite!(capacity, "capacity", x, y, _step);

                    // TODO: Small refactor to de-dupe.
                    if cell.suspended_load < capacity {
                        // erode
                        let diff   = capacity - cell.suspended_load;
                        let amount = KE * diff;
                        ensure_finite!(amount, "erosion", x, y, _step);
                        cell.elevation      -= amount;
                        cell.suspended_load += amount;
                    } else {
                        // deposit
                        let diff   = cell.suspended_load - capacity;
                        let amount = (KD * diff).min(MAX_ELEVATION - cell.elevation);
                        ensure_finite!(amount, "deposition", x, y, _step);
                        cell.elevation      += amount;
                        cell.suspended_load -= amount;
                    }
                }
            }
        }

        // 4b) Angle-of-repose adjustment (slope limit) – no fresh allocations
        enforce_angle_of_repose(hex_map, &mut delta_elev);

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

            render_frame(hex_map, frame_buffer, river_y);
            save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
            save_png("terrain.png", hex_map);

            println!(
                "Round {:.0}: water in {:.1}  water out {:.1}  stored {:.0}  mean depth {:.2} ft  max depth {:.2} ft  wet {:} ({:.1}%)  sediment in {:.2}  sediment out {:.2}",
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

//---------------------------------------------------------------------
// Enforce that no cell is more than HEX_SIZE ft higher than any neighbour
//---------------------------------------------------------------------
fn enforce_angle_of_repose(hex_map: &mut Vec<Vec<Hex>>, delta: &mut Vec<Vec<f32>>) {
    let height = hex_map.len();
    let width = hex_map[0].len();

    // zero the delta buffer
    delta.par_iter_mut().for_each(|row| row.fill(0.0));

    for y in 0..height {
        for x in 0..width {
            let elev = hex_map[y][x].elevation;
            ensure_finite!(elev, "elevation", x, y, -1);
            let offsets = if (x & 1) == 0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
            for &(dx, dy) in offsets {
                let nx = x as i32 + dx as i32;
                let ny = y as i32 + dy as i32;
                if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                    continue;
                }
                let nelev = hex_map[ny as usize][nx as usize].elevation;
                ensure_finite!(nelev, "neighbour elevation", nx as usize, ny as usize, -1);
                let diff = elev - nelev;
                if diff > HEX_SIZE {
                    // Divide by 7 to avoid nonsense if a hex receives "rockslides"
                    // from multiple neighbours.
                    let excess = (diff - HEX_SIZE) / 7.0;
                    delta[y][x] -= excess;
                    delta[ny as usize][nx as usize] += excess;
                }
            }
        }
    }

    // Apply deltas in a separate pass
    for y in 0..height {
        for x in 0..width {
            let d = delta[y][x];
            if d != 0.0 {
                hex_map[y][x].elevation += d;
                ensure_finite!(hex_map[y][x].elevation, format!("elevation (after delta {d})"), x, y, -1);
            }
        }
    }
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
    simulate_rainfall(&mut hex_map, total_steps, river_y, &mut frame_buffer);

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