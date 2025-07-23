use rand::Rng;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use rayon::prelude::*;
use image::{RgbImage};

// const HEIGHT_PIXELS: u16 = 2160;
// const WIDTH_PIXELS: u16 = 3840;
const HEIGHT_PIXELS: u16 = 72;
const WIDTH_PIXELS: u16 = 128;

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
const WIDTH_HEXAGONS: u16 = 148;

// This combined wtih some other things keeps the starting coastline in the middle of the map.
const SEA_LEVEL: f32 = (WIDTH_HEXAGONS as f32) * 4.0;
// This can be used to convert volume from foot-hexes to cubic feet.
const HEX_SIZE: f32 = 2640.0; // Feet
// const FAR_SOUTH_RANGE_START: u16 = 1420;

struct Hex {
    coordinate: (u16, u16),
    elevation: f32, // Feet
    water_depth: f32, // Feet of water currently stored in this hex
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
        // TODO: We might want to replace this with something that can be easily distinguished from actual indicators
        // of water depth, say purple, so that we can see where there's dry land below sea level.
        // Water: varying shades of blue
        let depth = SEA_LEVEL - elevation;
        let max_depth = SEA_LEVEL; // Assuming minimum elevation is 0
        let normalized_depth = (depth / max_depth).min(1.0);
        
        // Create a smooth blue gradient from light blue (near coast) to dark blue (deep water)
        // Invert the depth so shallow water is lighter
        let shallow_factor = 1.0 - normalized_depth;
        let blue_intensity = (100.0 + 155.0 * shallow_factor) as u8;
        let green_intensity = (100.0 + 50.0 * shallow_factor) as u8;
        let red_intensity = (50.0 + 20.0 * shallow_factor) as u8;
        
        Rgb([red_intensity, green_intensity, blue_intensity])
    } else {
        // Land: green -> yellow -> orange -> red -> brown -> white
        let land_height = elevation - SEA_LEVEL;
        let max_land_height = max_elevation - SEA_LEVEL;
        let normalized_height = (land_height as f32 / max_land_height as f32).min(1.0);
        
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
    rain_per_step: f32,
    river_depth_per_step: f32,
    river_y: usize,
    frame_buffer: &mut Vec<u32>,
    max_elevation: f32,
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

    for _step in 0..steps {
        // Mass balance stats per step
        let rainfall_added = (width * height) as f32 * rain_per_step;
        let mut river_added = 0.0f32;
        let mut step_outflow = 0.0f32;

        // 1) Add rainfall uniformly (parallel over rows)
        hex_map.par_iter_mut().for_each(|row| {
            for hex in row {
                hex.water_depth += rain_per_step;
            }
        });

        // 1b) Add river inflow at east edge (x = WIDTH_HEXAGONS-1)
        if river_y < height {
            hex_map[river_y][WIDTH_HEXAGONS as usize - 1].water_depth += river_depth_per_step;
            river_added = river_depth_per_step;
        }

        // 2) Clear reusable next_water buffer in parallel
        next_water.par_iter_mut().for_each(|row| row.fill(0.0));

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
                        if diff / 2.0 > w {
                            // Avoid moving more water than is available.
                            next_water[ty][tx] += w;
                        } else {
                            // Attempt to equalize the water levels of the two hexes.
                            next_water[ty][tx] += diff / 2.0;
                            next_water[y][x] += w - diff / 2.0;
                        }
                    }
                    None => {
                        // No lower neighbour; water stays put
                        next_water[y][x] += w;
                    }
                }
            }
        }

        // 4) Apply next water depths, counting outflow at sea boundary (x == 0)
        for y in 0..height {
            for x in 0..width {
                let new_w = next_water[y][x];
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
                } else {
                    hex_map[y][x].water_depth = new_w;
                }
            }
        }

        if _step % 1000 == 0 {
            let (water_on_land, max_depth) = hex_map
                .par_iter()
                .map(|row| {
                    let mut sum = 0.0f32;
                    let mut row_max = 0.0f32;
                    for h in row {
                        let d = h.water_depth;
                        sum += d;
                        if d > row_max {
                            row_max = d;
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

            let mean_depth = water_on_land / (width * height) as f32;
            let water_on_land_in_feet = water_on_land; // * HEX_SIZE * HEX_SIZE * HEX_FACTOR;

            let wet_cells: usize = hex_map
                .par_iter()
                .map(|row| row.iter().filter(|h| h.water_depth > 0.05).count())
                .sum();

            render_frame(hex_map, frame_buffer, river_y, max_elevation);

            println!(
                "Step {:>3}: total rain + inflow {:.0} ft-hexes  outflow {:.0} ft-hexes  on land {:.0} ft-hexes  mean {:.3} ft  max {:.2} ft  wet cells {}",
                _step,
                rainfall_added + river_added,
                step_outflow,
                water_on_land_in_feet,
                mean_depth,
                max_depth,
                wet_cells
            );
        }
    }

    // Get final frame so it can be saved later.
    render_frame(hex_map, frame_buffer, river_y, max_elevation);

    let water_remaining: f32 = hex_map
        .iter()
        .flat_map(|row| row.iter())
        .map(|h| h.water_depth)
        .sum();

    let water_remaining_in_feet = water_remaining; // * HEX_SIZE * HEX_SIZE * HEX_FACTOR;

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft-hexes, water remaining on land: {:.2} ft-hexes",
        steps, total_outflow, water_remaining_in_feet
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
    let _ = img.save(path);
}

// Renders current hex_map state into an RGB buffer (u32 per pixel)
fn render_frame(hex_map: &Vec<Vec<Hex>>, buffer: &mut [u32], river_y: usize, max_elevation: f32) {
    for y in 0..HEIGHT_PIXELS {
        for x in 0..WIDTH_PIXELS {
            let hex_x = ((x as f32) / HEX_FACTOR) as u16;
            let hex_y = y;

            let hex_x = hex_x.min(WIDTH_HEXAGONS - 1);
            let hex_y = hex_y.min(HEIGHT_PIXELS - 1);

            let hex = &hex_map[hex_y as usize][hex_x as usize];
            // Choose colour – highlight water depth strongly so it stands out
            let color = if hex.water_depth > 0.05 {
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

    let neighbors = hex_neighbors((0, 0));
    println!("{:?}", neighbors);

    // Other possible test cases for this logic:
    // let mut neighbors = hex_neighbors((0, 1));
    // println!("{:?}", neighbors);
    // let mut neighbors = hex_neighbors((1, 0));
    // println!("{:?}", neighbors);
    // let mut neighbors = hex_neighbors((1, 1));
    // println!("{:?}", neighbors);

    // Time hex map creation
    let hex_start = Instant::now();
    for y in 0..HEIGHT_PIXELS {
        hex_map.push(Vec::new());
        for x in 0..WIDTH_HEXAGONS {
            let x_based_elevation = (x as f32) * 4.0 + SEA_LEVEL / 2.0;
            // let mut far_south_bonus = 0;
            // TODO: Revisit this. The idea was to have a mountian range at the south end of the map, but I found
            // my initial attempts unsatisfying.
            // if y > FAR_SOUTH_RANGE_START {
            //     far_south_bonus = (y - FAR_SOUTH_RANGE_START) * 12;
            //     if x == 0 {
            //         println!("x_based_elevation: {}, far_south_bonus: {}", x_based_elevation, far_south_bonus);
            //     }
            // }
            let elevation = x_based_elevation + (rng.gen_range(0..24) as f32);
            let mut water_depth = 0.0;
            // This allows for the possibility of pockets of dry land below sea level. It errs on the side of
            // starting with zero water, I could probably do something fancier with pathfinding to guarantee
            // hexes start with water IFF they have a path to the sea.
            if x_based_elevation + 24.0 < SEA_LEVEL {
                water_depth = SEA_LEVEL - elevation;
            }
            hex_map[y as usize].push(Hex {
                coordinate: (x, y),
                elevation: elevation,
                water_depth: water_depth,
            });
        }
    }

    let max_elevation = hex_map
        .par_iter()                                       // rows in parallel
        .map(|row| row.iter().map(|h| h.elevation).fold(f32::NEG_INFINITY, f32::max))
        .reduce(|| f32::NEG_INFINITY, f32::max);

    // --------------------------------------------------------
    // Milestone 1: pure rainfall with fixed sea boundary
    // --------------------------------------------------------

    let mut frame_buffer = vec![0u32; (WIDTH_PIXELS as usize) * (HEIGHT_PIXELS as usize)];

    simulate_rainfall(&mut hex_map, (WIDTH_HEXAGONS as u32) * 1000, 0.0, 5.0, 21, &mut frame_buffer, max_elevation);

    // Count final blue pixels for quick sanity check
    let final_blue = frame_buffer
        .iter()
        .filter(|&&px| (px & 0x0000FF) == 0x0000FF && (px >> 16 & 0xFF) == 0 && (px >> 8 & 0xFF) == 0)
        .count();
    println!("Final blue pixels: {}", final_blue);

    save_buffer_png("terrain_water.png", &frame_buffer, WIDTH_PIXELS as u32, HEIGHT_PIXELS as u32);

    let hex_duration = hex_start.elapsed();
    println!("Hex map creation took: {:?}", hex_duration);

    // Time PNG conversion
    let png_start = Instant::now();
    
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
            let color = elevation_to_color(elevation, max_elevation);
            
            // Set the pixel
            img.put_pixel(x as u32, y as u32, color);
        }
    }

    let mut map_corners = Vec::new();
    map_corners.push(hex_map[0][0].elevation);
    map_corners.push(hex_map[0][WIDTH_HEXAGONS as usize - 1].elevation);
    map_corners.push(hex_map[HEIGHT_PIXELS as usize - 1][0].elevation);
    map_corners.push(hex_map[HEIGHT_PIXELS as usize - 1][WIDTH_HEXAGONS as usize - 1].elevation);
    println!("{:?}", map_corners);
    
    let png_conversion_duration = png_start.elapsed();
    println!("PNG conversion took: {:?}", png_conversion_duration);
    
    // Time file saving separately
    // TODO: This is slow, should just pop up in window rather than worry about saving.
    let save_start = Instant::now();
    img.save("terrain.png").expect("Failed to save image");
    let save_duration = save_start.elapsed();
    println!("File saving took: {:?}", save_duration);
    println!("Terrain visualization saved as terrain.png");
}