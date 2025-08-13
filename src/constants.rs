pub const ONE_MILE: f32 = 5280.0;
pub const HEX_SIZE: f32 = ONE_MILE / 2.0;
// Randomly create slopes up to 1% grade.
pub const RANDOM_ELEVATION_FACTOR: f32 = HEX_SIZE * 0.01;
// sqrt(3) / 2
pub const HEX_FACTOR: f32 = 0.86602540378;
// TODO: Are there any places this would be useful?
pub const HEX_WIDTH: f32 = HEX_SIZE * HEX_FACTOR;

pub const HEIGHT_PIXELS: usize = 2160;
pub const WIDTH_PIXELS: usize = 3840;
pub const WIDTH_HEXAGONS: usize = (WIDTH_PIXELS as f32 / HEX_FACTOR) as usize;

// Use for inland north desert.
pub const VERY_LOW_RAIN: f32 = 1.0;
// Used for coastal north desert and inland central highland.
pub const LOW_RAIN: f32 = 10.0;
// Used for coastal central highland and inland south mountains.
pub const MEDIUM_RAIN: f32 = 21.0;
// Used for coastal south mountains.
pub const HIGH_RAIN: f32 = 34.0;
// Used for NE basin.
pub const VERY_HIGH_RAIN: f32 = 49.0;
// Above numbers are in inches per year, this can be adjusted to e.g. feet per year.
pub const RAINFALL_FACTOR: f32 = 1.0 / 12.0 / 365.0 / 24.0 / 6.0;
// Some dubiously realistic back-of-the-envelope math suggests maybe I should use 16.0 here,
// but maybe I forgot a unit conversion? A multipler of about 500x is the absolute limit
// before too little water from the main river will reach the sea.
pub const EVAPORATION_FACTOR: f32 = RAINFALL_FACTOR * 16.0;

pub const ONE_DEGREE_LATITUDE_MILES: f32 = 69.05817;
pub const RIVER_Y: usize = (4.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NORTH_DESERT_HEIGHT: usize = (6.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NE_BASIN_HEIGHT: usize = NORTH_DESERT_HEIGHT;
pub const CENTRAL_HIGHLAND_HEIGHT: usize = (11.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize - NORTH_DESERT_HEIGHT;
pub const SOUTH_MOUNTAINS_HEIGHT: usize = HEIGHT_PIXELS - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT;

// 800 miles in hexes.
pub const MAIN_RIVER_WIDTH: usize = (800.0 * 2.0 / HEX_FACTOR) as usize;
// Delta will grow over time but we are "seeding" an area for it to form.
// May need to put hills north and south of RIVER_Y, DELTA_SEED_WIDTH from
// the coast, to force the exact location, but will test without first.
// TODO: Remove once we're sure we don't need it.
pub const DELTA_SEED_WIDTH: usize = 0;
// River runs through north desert
pub const NORTH_DESERT_WIDTH: usize = MAIN_RIVER_WIDTH + DELTA_SEED_WIDTH;
pub const COAST_WIDTH: usize = (72.0 * 2.0 / HEX_FACTOR) as usize;
pub const NE_BASIN_FRINGE: usize = 4;
pub const NORTH_DESERT_RAIN: f32 = (COAST_WIDTH as f32 * LOW_RAIN + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * VERY_LOW_RAIN) * NORTH_DESERT_HEIGHT as f32;
// Rain on the part of the central highland whose east-west extent corresponds to the north desert.
pub const MAIN_CENTRAL_HIGHLAND_RAIN: f32 = (COAST_WIDTH as f32 * MEDIUM_RAIN + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * LOW_RAIN) * CENTRAL_HIGHLAND_HEIGHT as f32;
// An attempt to balance water in north and central regions mathematically.
pub const NE_BASIN_WIDTH: usize = ((MAIN_CENTRAL_HIGHLAND_RAIN - NORTH_DESERT_RAIN) / (VERY_HIGH_RAIN * NE_BASIN_HEIGHT as f32)) as usize;
pub const TOTAL_LAND_WIDTH: usize = NE_BASIN_WIDTH + NORTH_DESERT_WIDTH;
// In real life, continental shelves can extend as little as 50 miles or as much as 310 miles from shore.
pub const CONTINENTAL_SHELF_WIDTH: usize = (230.0 * 2.0 / HEX_FACTOR) as usize;
pub const CONTINENTAL_SHELF_DEPTH: f32 = 460.0;
pub const CONTINENTAL_SHELF_INCREMENT: f32 = CONTINENTAL_SHELF_DEPTH as f32 / CONTINENTAL_SHELF_WIDTH as f32;
pub const CONTINENTAL_SLOPE_GRADE: f32 = 0.05; // About 3 degrees.
pub const CONTINENTAL_SLOPE_INCREMENT: f32 = CONTINENTAL_SLOPE_GRADE * HEX_SIZE * HEX_FACTOR;
pub const ABYSSAL_PLAINS_MIN_DEPTH: f32 = 10_000.0;
pub const ABYSSAL_PLAINS_MAX_DEPTH: f32 = 15_000.0;
pub const CONTINENTAL_SLOPE_WIDTH: usize = ((ABYSSAL_PLAINS_MIN_DEPTH - CONTINENTAL_SHELF_DEPTH) / CONTINENTAL_SLOPE_INCREMENT) as usize;
pub const TOTAL_SEA_WIDTH: usize = WIDTH_HEXAGONS - TOTAL_LAND_WIDTH;
pub const RIVER_SOURCE_X: usize = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE + 1;
pub const ABYSSAL_PLAINS_WIDTH: usize = TOTAL_SEA_WIDTH - CONTINENTAL_SHELF_WIDTH - CONTINENTAL_SLOPE_WIDTH;
pub const ABYSSAL_PLAINS_INCREMENT: f32 = (ABYSSAL_PLAINS_MAX_DEPTH - ABYSSAL_PLAINS_MIN_DEPTH) / ABYSSAL_PLAINS_WIDTH as f32;
pub const SEA_LEVEL: f32 = 0.0;

pub const NORTH_DESERT_MAX_ELEVATION: f32 = 8_000.0;

// With Perlin noise, actual elevation will likely be lower than these numbers.
pub const CENTRAL_HIGHLAND_MAX_ELEVATION: f32 = 12_000.0;
pub const SOUTH_MOUNTAINS_MAX_ELEVATION: f32 = 20_000.0;

pub const KC: f32 = 1.5; // capacity coefficient
pub const KE: f32 = 0.15; // erosion rate fraction
// TODO: experimentally still get the weird little pits in lakes at 0.05
// got reduced number of weird little pits at 0.02, let's try 0.01
pub const KD: f32 = 0.01; // deposition rate fraction

// TODO: Remove constants associated with the hard-coded river source after fully replacing it with the NE basin.
// 0.01 hours of average flow through the Aswan Dam.
pub const RIVER_WATER_PER_STEP: f32 = 0.371; // Feet
pub const FLOW_FACTOR: f32 = 0.90;
// Might take 7k-10k rounds to carve out the river valley I want.
pub const DEFAULT_ROUNDS: u32 = 4_800;
pub const WATER_THRESHOLD: f32 = 1.0 / 12.0; // Feet


pub const MAX_SLOPE: f32 = 1.00;
pub const MAX_FLOW: f32 = (HEX_SIZE as f32) * MAX_SLOPE;
// Current highest of all max elevation constants.
pub const MAX_ELEVATION: f32 = SEA_LEVEL + SOUTH_MOUNTAINS_MAX_ELEVATION;
pub const LOG_ROUNDS: u32 = 100;

pub const BIG_VOLCANO_X: usize = TOTAL_SEA_WIDTH + DELTA_SEED_WIDTH + (1_000.0 / HEX_FACTOR) as usize;

// TODO: Configure island chain to be about 500 miles off the coast.