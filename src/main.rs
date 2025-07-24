use rand::Rng;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use rayon::prelude::*;
use image::{RgbImage};

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
const SEA_LEVEL: f32 = (WIDTH_HEXAGONS as f32) * 4.0;
// This can be used to convert volume from foot-hexes to cubic feet.
const HEX_SIZE: f32 = 2640.0; // Feet
// One hour worth of average discharge from the Aswan Dam.
const RIVER_DEPTH_PER_STEP: f32 = 37.1; // Feet
// Basically desert conditions
const RAIN_PER_STEP: f32 = 0.000_003_08 * RIVER_DEPTH_PER_STEP; // Feet
const STEP_MULTIPLIER: u32 = 1000;
const WATER_THRESHOLD: f32 = 1.0 / 12.0; // One inch in feet

// --- Minimal erosion / deposition constants ---
// Questioning the decision to divide by HEX_SIZE in errosion calculations.
const KC: f32 = HEX_SIZE / 100.0; // capacity coefficient
const KE: f32 = 0.1;  // erosion rate fraction
const KD: f32 = 0.1;  // deposition rate fraction

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

fn elevation_to_color(elevation: f32, max_elevation: f32) -> Rgb<u8> {
    if elevation < SEA_LEVEL {
        let normalized_elevation = (elevation / SEA_LEVEL).min(1.0);
        if normalized_elevation < 0.5 {
            // Clearly mark where errosion has lowered elevation below the lowest possible initial elevation.
            Rgb([0, 0, 0])
        } else if normalized_elevation < 0.75 {
            // Black to purple
            let factor = (normalized_elevation - 0.5) / 0.25;
            let red = 128 * factor as u8;
            let green = 0;
            let blue = 128 * factor as u8;
            Rgb([red, green, blue])
        } else {
            // Purple to light blue
            let factor = (normalized_elevation - 0.75) / 0.25;
            let red = 128;
            let green = 255 * factor as u8;
            let blue = 128 + (127.0 * factor) as u8;
            Rgb([red, green, blue])
        }
    } else {
        // Land: green -> yellow -> orange -> red -> brown -> white
        let land_height = elevation - SEA_LEVEL;
        let max_land_height = max_elevation - SEA_LEVEL;
        let normalized_height = (land_height / max_land_height).min(1.0);
        
        if normalized_height < 0.225 {
            // Green to yellow
            let factor = normalized_height / 0.225;
            let green = 255;
            let red = (255.0 * factor) as u8;
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

    let mut total_outflow = 0.0f32;

    // Reusable buffer for next water depths
    let mut next_water: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    // Reusable buffer for sediment transport (same layout)
    let mut next_load: Vec<Vec<f32>> = (0..height)
        .map(|_| vec![0.0f32; width])
        .collect();

    for _step in 0..steps {
        // Mass balance stats per step
        let rainfall_added = (width * height) as f32 * RAIN_PER_STEP;
        let river_added = 0.0f32;
        let mut step_outflow = 0.0f32;

        let mut step_eroded    = 0.0f32;   // total bed material removed this step
        let mut step_deposited = 0.0f32;   // total bed material deposited

        // 1) Add rainfall uniformly (parallel over rows)
        hex_map.par_iter_mut().for_each(|row| {
            for hex in row {
                // TODO: Add some randomness to the rainfall.
                hex.water_depth += RAIN_PER_STEP;
            }
        });

        // 1b) Add river inflow at east edge (x = WIDTH_HEXAGONS-1)
        if river_y < height {
            hex_map[river_y][WIDTH_HEXAGONS as usize - 1].water_depth += RIVER_DEPTH_PER_STEP;

            let suspended_load_per_step = RIVER_DEPTH_PER_STEP * 0.427 * KC / HEX_SIZE;
            hex_map[river_y][WIDTH_HEXAGONS as usize - 1].suspended_load += suspended_load_per_step;
            step_eroded += suspended_load_per_step;
        }

        // 2) Clear reusable next_water buffer in parallel
        next_water.par_iter_mut().for_each(|row| row.fill(0.0));
        next_load.par_iter_mut().for_each(|row| row.fill(0.0));

        // 3) Route water once (sequential for now – write conflicts are tricky to parallelise safely)
        for y in 0..height {
            for x in 0..width {
                let cell = &hex_map[y][x];
                let w = cell.water_depth;
                if w <= 0.0 {
                    continue;
                }

                let mut min_height = cell.elevation as f32 + cell.water_depth;
                let mut target: Option<(usize, usize)> = None;

                let neighbours = hex_neighbors((x as u16, y as u16));
                for (nx, ny) in neighbours {
                    let n_hex = &hex_map[ny as usize][nx as usize];
                    let neighbour_height = n_hex.elevation + n_hex.water_depth;
                    if neighbour_height < min_height {
                        min_height = neighbour_height;
                        target = Some((nx as usize, ny as usize));
                    }
                }

                // TODO: In this version of the code, we're moving either all water, or no water,
                // will test this to see how it works but could lead to some strange behavior.
                match target {
                    Some((tx, ty)) => {
                        let target_hex = &hex_map[ty][tx];
                        let diff = cell.elevation + w - (target_hex.elevation + target_hex.water_depth);
                        if diff > w {
                            // Avoid moving more water than is available.
                            next_water[ty][tx] += w;
                            next_load[ty][tx] += cell.suspended_load; // move all load with water
                        } else {
                            // Attempt to equalize the water levels of the two hexes.
                            next_water[ty][tx] += diff / 2.0;
                            next_water[y][x] += w - diff / 2.0;
                            let load_move = cell.suspended_load * (diff / 2.0) / w;
                            next_load[ty][tx] += load_move;
                            next_load[y][x] += cell.suspended_load - load_move;
                        }
                    }
                    None => {
                        // No lower neighbour; water stays put
                        next_water[y][x] += w;
                        next_load[y][x] += cell.suspended_load;
                    }
                }
            }
        }

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
                    hex_map[y][x].suspended_load = 0.0; // flushed to sea
                } else {
                    // First gather immutable info
                    let cell_elev = hex_map[y][x].elevation;
                    let mut min_n = cell_elev;
                    for (nx, ny) in hex_neighbors((x as u16, y as u16)) {
                        let n_elev = hex_map[ny as usize][nx as usize].elevation;
                        if n_elev < min_n {
                            min_n = n_elev;
                        }
                    }

                    // Now mutate cell safely
                    let cell = &mut hex_map[y][x];
                    cell.water_depth = new_w;
                    cell.suspended_load = new_load;

                    let slope = ((cell_elev - min_n) / HEX_SIZE).max(0.0);
                    let capacity = KC * cell.water_depth * slope;

                    if cell.suspended_load < capacity {
                        // erode
                        let diff   = capacity - cell.suspended_load;
                        let amount = KE * diff;
                        cell.elevation      -= amount;
                        cell.suspended_load += amount;
                        step_eroded         += amount;
                    } else {
                        // deposit
                        let diff   = cell.suspended_load - capacity;
                        let amount = KD * diff;
                        cell.elevation      += amount;
                        cell.suspended_load -= amount;
                        step_deposited      += amount;
                    }
                }
            }
        }

        // 4b) Angle-of-repose adjustment (slope limit)
        enforce_angle_of_repose(hex_map);

        if _step % (WIDTH_HEXAGONS as u32) == 0 {
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

            render_frame(hex_map, frame_buffer, river_y);
            save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
            save_png("terrain.png", hex_map);

            println!(
                "Step {:>5}: rain+river {:.1}  outflow {:.1}  stored {:.0}  mean {:.2} ft  max {:.2} ft  wet {:} ({:.1}%)  erod {:.3}  dep {:.3}",
                _step,
                (rainfall_added + river_added),
                step_outflow,
                water_on_land,
                mean_depth,
                max_depth,
                wet_cells,
                wet_cells_percentage,
                step_eroded,
                step_deposited,
            );
        }
    }

    let water_remaining: f32 = hex_map
        .iter()
        .flat_map(|row| row.iter())
        .map(|h| h.water_depth)
        .sum();

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft-hexes, water remaining on land: {:.2} ft-hexes",
        steps, total_outflow, water_remaining
    );
}

//---------------------------------------------------------------------
// Enforce that no cell is more than HEX_SIZE ft higher than any neighbour
//---------------------------------------------------------------------
fn enforce_angle_of_repose(hex_map: &mut Vec<Vec<Hex>>) {
    let height = hex_map.len();
    let width = hex_map[0].len();

    // temp elevation changes
    let mut delta: Vec<Vec<f32>> = vec![vec![0.0; width]; height];

    for y in 0..height {
        for x in 0..width {
            let elev = hex_map[y][x].elevation;
            for (nx, ny) in hex_neighbors((x as u16, y as u16)) {
                let nelev = hex_map[ny as usize][nx as usize].elevation;
                let diff = elev - nelev;
                if diff > HEX_SIZE {
                    let excess = (diff - HEX_SIZE) / 2.0; // move half each way
                    delta[y][x]      -= excess;
                    delta[ny as usize][nx as usize] += excess;
                }
            }
        }
    }

    // Apply deltas
    for y in 0..height {
        for x in 0..width {
            hex_map[y][x].elevation += delta[y][x];
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

fn get_max_elevation(hex_map: &Vec<Vec<Hex>>) -> f32 {
    hex_map
        .par_iter()                                       // rows in parallel
        .map(|row| row.iter().map(|h| h.elevation).fold(f32::NEG_INFINITY, f32::max))
        .reduce(|| f32::NEG_INFINITY, f32::max)
}

fn save_png(path: &str, hex_map: &Vec<Vec<Hex>>) {
    // Create the visualization image
    let mut img = ImageBuffer::new(WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);
    
    let max_elevation = get_max_elevation(hex_map);

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
            let color = elevation_to_color(elevation, max_elevation);
            
            // Set the pixel
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    
    img.save(path).expect("Failed to save image");
}

// Renders current hex_map state into an RGB buffer (u32 per pixel)
fn render_frame(hex_map: &Vec<Vec<Hex>>, buffer: &mut [u32], _river_y: usize) {
    let max_elevation = get_max_elevation(hex_map);

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
                let Rgb([r, g, b]) = elevation_to_color(hex.elevation, max_elevation);
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

            let x_based_elevation = (x as f32) * 4.0 * southern_multiplier + SEA_LEVEL / 2.0;

            let mut elevation = x_based_elevation;
            if y == 0 || y == HEIGHT_PIXELS - 1 || x == WIDTH_HEXAGONS - 1 {
                // Try to prevent weird artifacts at the edges of the map. Not applied to western edge because
                // we have the outflow mechanic there.
                elevation += rng.gen_range(HEX_SIZE/200.0..HEX_SIZE/100.0);
            } else {
                elevation += rng.gen_range(0.0..HEX_SIZE/100.0);
            }
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

    // --------------------------------------------------------
    // Milestone 1: pure rainfall with fixed sea boundary
    // --------------------------------------------------------

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    // River silt should be 0.00245 * river water
    // Or maybe 0.427 * KC * river water? (Idea is to have incoming water be saturated with silt assuming slope is a little over 0.4 feet per hex)
    // Maybe KC should be around 0.00574 to harmonize these?
    simulate_rainfall(&mut hex_map, (WIDTH_HEXAGONS as u32) * STEP_MULTIPLIER, river_y, &mut frame_buffer);

    // Count final blue pixels for quick sanity check
    let final_blue = frame_buffer
        .iter()
        .filter(|&&px| (px & 0x0000FF) == 0x0000FF && (px >> 16 & 0xFF) == 0 && (px >> 8 & 0xFF) == 0)
        .count();
    println!("Final blue pixels: {}", final_blue);

    render_frame(&mut hex_map, &mut frame_buffer, river_y);
    save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

    let hex_duration = hex_start.elapsed();
    println!("Hex map creation took: {:?}", hex_duration);

    // Time PNG conversion
    let png_start = Instant::now();
    save_png("terrain.png", &hex_map);

    let save_duration = png_start.elapsed();
    println!("File saving took: {:?}", save_duration);
    println!("Terrain visualization saved as terrain.png");
}