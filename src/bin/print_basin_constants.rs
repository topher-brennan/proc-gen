// Simple script to print NE_BASIN_HEIGHT and NE_BASIN_WIDTH
// These values are calculated from constants defined in constants.rs

fn main() {
    // Constants needed for calculation
    const ONE_DEGREE_LATITUDE_MILES: f32 = 69.05817;
    const HEX_FACTOR: f32 = 0.86602540378;

    // Rainfall constants
    const VERY_LOW_RAIN: f32 = 1.0;
    const LOW_RAIN: f32 = 10.0;
    const MEDIUM_RAIN: f32 = 21.0;
    const VERY_HIGH_RAIN: f32 = 49.0;

    // Calculate derived constants
    const MIN_NORTH_DESERT_HEIGHT: usize = (6.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
    const NE_BASIN_HEIGHT: usize = MIN_NORTH_DESERT_HEIGHT;

    const MAIN_RIVER_WIDTH: usize = (800.0 * 2.0 / HEX_FACTOR) as usize;
    const DELTA_SEED_WIDTH: usize = 0;
    const NORTH_DESERT_WIDTH: usize = MAIN_RIVER_WIDTH + DELTA_SEED_WIDTH;
    const COAST_WIDTH: usize = (72.0 * 2.0 / HEX_FACTOR) as usize;

    // Calculate CENTRAL_HIGHLAND_HEIGHT
    const CENTRAL_HIGHLAND_HEIGHT: usize =
        (11.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize - MIN_NORTH_DESERT_HEIGHT;

    // Calculate rain values
    const NORTH_DESERT_RAIN: f32 = (COAST_WIDTH as f32 * LOW_RAIN
        + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * VERY_LOW_RAIN)
        * MIN_NORTH_DESERT_HEIGHT as f32;
    const MAIN_CENTRAL_HIGHLAND_RAIN: f32 = (COAST_WIDTH as f32 * MEDIUM_RAIN
        + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * LOW_RAIN)
        * CENTRAL_HIGHLAND_HEIGHT as f32;

    // Calculate NE_BASIN_WIDTH
    const NE_BASIN_WIDTH: usize = ((MAIN_CENTRAL_HIGHLAND_RAIN - NORTH_DESERT_RAIN)
        / (VERY_HIGH_RAIN * NE_BASIN_HEIGHT as f32)) as usize;

    println!("NE_BASIN_HEIGHT: {}", NE_BASIN_HEIGHT);
    println!("NE_BASIN_WIDTH: {}", NE_BASIN_WIDTH);
}
