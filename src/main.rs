use rand::Rng;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use std::cmp::max;

const HEIGHT_PIXELS: u16 = 2160;
const WIDTH_PIXELS: u16 = 3840;

// Approximate value of sqrt(3) / 2
// Useful because if the length of a perpendicular line segment connecting two
// sides of a regular hexagon is 1, then the length of a line segment
// connecting the corners of the hexagon is 2 / sqrt(3), and the side length is
// 1 / sqrt(3). These values are equal to 2 * sqrt(3) / 3 and sqrt(3) / 3,
// respectively, so their average is ([2 + 1] * sqrt(3)) / 3 / 2 = sqrt(3) / 2
// This is the area and average width of the hexagon.
const HEX_FACTOR: f32 = 0.8660254037844386;

// Assume 1 pixel per hex vertically, and HEX_FACTOR pixels per hex horizontally.
const WIDTH_HEXAGONS: u16 = 4434;

const SEA_LEVEL: u16 = 17736; // Feet
const FAR_SOUTH_RANGE_START: u16 = 1420;

struct Hex {
    coordinate: (u16, u16),
    elevation: u16, // Feet
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

fn elevation_to_color(elevation: u16) -> Rgb<u8> {
    if elevation < SEA_LEVEL {
        // Water: varying shades of blue
        let depth = SEA_LEVEL - elevation;
        let max_depth = SEA_LEVEL; // Assuming minimum elevation is 0
        let normalized_depth = (depth as f32 / max_depth as f32).min(1.0);
        
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
        let max_land_height = SEA_LEVEL * 2; // Adjust this based on your terrain range
        let normalized_height = (land_height as f32 / max_land_height as f32).min(1.0);
        
        if normalized_height < 0.2 {
            // Green to yellow
            let factor = normalized_height / 0.2;
            let green = 255;
            let red = (255.0 * factor) as u8;
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.4 {
            // Yellow to orange
            let factor = (normalized_height - 0.2) / 0.2;
            let red = 255;
            let green = (255.0 * (1.0 - factor)) as u8;
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.6 {
            // Orange to red
            let factor = (normalized_height - 0.4) / 0.2;
            let red = 255;
            let green = 0;
            let blue = 0;
            Rgb([red, green, blue])
        } else if normalized_height < 0.8 {
            // Red to brown
            let factor = (normalized_height - 0.6) / 0.2;
            let red = 255;
            let green = (100.0 * factor) as u8;
            let blue = (50.0 * factor) as u8;
            Rgb([red, green, blue])
        } else {
            // Brown to white
            let factor = (normalized_height - 0.8) / 0.2;
            let intensity = (200.0 + 55.0 * factor) as u8;
            Rgb([intensity, intensity, intensity])
        }
    }
}

// TODO: Vary rainfall so that there's more rainfall further south, and more rainfall closer to the coast.
// TODO: Visualize results.
// ------------------------------------------------------------
// Simple rainfall–runoff experiment (Milestone 1)
// ------------------------------------------------------------
fn simulate_rainfall(hex_map: &mut Vec<Vec<Hex>>, steps: u32, rain_per_step: f32) {
    let height = hex_map.len();
    if height == 0 {
        return;
    }
    let width = hex_map[0].len();

    let mut total_outflow = 0.0f32;

    for _step in 0..steps {
        println!("Step: {}", _step);
        // 1) Add rainfall uniformly
        for row in hex_map.iter_mut() {
            for hex in row.iter_mut() {
                hex.water_depth += rain_per_step;
            }
        }

        // 2) Route water once (instantaneous transfer to lowest neighbour)
        let mut next_water: Vec<Vec<f32>> = vec![vec![0.0; width]; height];

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
                    let neighbour_height = n_hex.elevation as f32 + n_hex.water_depth;
                    if neighbour_height < min_height {
                        min_height = neighbour_height;
                        target = Some((nx as usize, ny as usize));
                    }
                }

                // TODO: In this version of the code, we're moving either all water, or no water,
                // will test this to see how it works but could lead to some strange behavior.
                match target {
                    Some((tx, ty)) => {
                        next_water[ty][tx] += w;
                    }
                    None => {
                        // No lower neighbour; water stays put
                        next_water[y][x] += w;
                    }
                }
            }
        }

        // 3) Apply next water depths, counting outflow at sea boundary (x == 0)
        for y in 0..height {
            for x in 0..width {
                let new_w = next_water[y][x];
                if x == 0 {
                    // Outflow to sea
                    total_outflow += new_w;
                    hex_map[y][x].water_depth = 0.0;
                } else {
                    hex_map[y][x].water_depth = new_w;
                }
            }
        }
    }

    let water_remaining: f32 = hex_map
        .iter()
        .flat_map(|row| row.iter())
        .map(|h| h.water_depth)
        .sum();

    println!(
        "Rainfall simulation complete – steps: {}, total outflow to sea: {:.2} ft, water remaining on land: {:.2} ft",
        steps, total_outflow, water_remaining
    );
}

fn main() {
    let mut hex_map = Vec::new();
    let mut rng = rand::thread_rng();

    let mut neighbors = hex_neighbors((0, 0));
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
            let x_based_elevation = x * 4 + SEA_LEVEL / 2;
            let mut far_south_bonus = 0;
            // TODO: Revisit this, currently finding it unsatisfying.
            // if y > FAR_SOUTH_RANGE_START {
            //     far_south_bonus = (y - FAR_SOUTH_RANGE_START) * 12;
            //     if x == 0 {
            //         println!("x_based_elevation: {}, far_south_bonus: {}", x_based_elevation, far_south_bonus);
            //     }
            // }
            hex_map[y as usize].push(Hex { coordinate: (x, y), elevation: x_based_elevation + far_south_bonus + rng.gen_range(0..48),
                water_depth: 0.0,
            });
        }
    }

    // --------------------------------------------------------
    // Milestone 1: pure rainfall with fixed sea boundary
    // --------------------------------------------------------
    simulate_rainfall(&mut hex_map, 100, 0.5);

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
            let color = elevation_to_color(elevation);
            
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