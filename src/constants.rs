pub const ONE_MILE: f32 = 5280.0;
pub const HEX_SIZE: f32 = ONE_MILE / 2.0;
// Randomly create slopes up to 1% grade.
pub const RANDOM_ELEVATION_FACTOR: f32 = HEX_SIZE * 0.01;
// sqrt(3) / 2
pub const HEX_FACTOR: f32 = 0.86602540378;
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
// Used for NE plateau.
pub const VERY_HIGH_RAIN: f32 = 49.0;
// Above numbers are in inches per year, this can be adjusted to e.g. feet per year.
pub const RAINFALL_FACTOR: f32 = 1.0 / 12.0 / 365.0 / 24.0 / 6.0;

pub const ONE_DEGREE_LATITUDE_MILES: f32 = 69.05817;
pub const RIVER_Y: usize = (4.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NORTH_DESERT_HEIGHT: usize = (6.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NE_PLATEAU_HEIGHT: usize = NORTH_DESERT_HEIGHT;
pub const CENTRAL_HIGHLAND_HEIGHT: usize = (11.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize - NORTH_DESERT_HEIGHT;
pub const SOUTH_MOUNTAINS_HEIGHT: usize = HEIGHT_PIXELS - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT;

// 800 miles in hexes.
pub const MAIN_RIVER_WIDTH: usize = (800.0 * 2.0 / HEX_FACTOR) as usize;
// Delta will grow over time but we are "seeding" an area for it to form.
// May need to put hills north and south of RIVER_Y, DELTA_SEED_WIDTH from
// the coast, to force the exact location, but will test without first.
pub const DELTA_SEED_WIDTH: usize = 200;
// River runs through north desert
pub const NORTH_DESERT_WIDTH: usize = MAIN_RIVER_WIDTH + DELTA_SEED_WIDTH;
pub const COAST_WIDTH: usize = (72.0 / HEX_FACTOR) as usize;
pub const NE_PLATEAU_FRINGE: usize = 4;
pub const NORTH_DESERT_RAIN: f32 = (COAST_WIDTH as f32 * LOW_RAIN + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * VERY_LOW_RAIN) * NORTH_DESERT_HEIGHT as f32;
// Rain on the part of the central highland whose east-west extent corresponds to the north desert.
pub const MAIN_CENTRAL_HIGHLAND_RAIN: f32 = (COAST_WIDTH as f32 * MEDIUM_RAIN + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * LOW_RAIN) * CENTRAL_HIGHLAND_HEIGHT as f32;
// Equalize total rain in central highland and area north of it (desert + plateau).
// Total central highland rain = MAIN_CENTRAL_HIGHLAND_RAIN + NE_PLATEAU_WIDTH * LOW_RAIN * CENTRAL_DESERT_HEIGHT
// Total rain north of central highland = NORTH_DESERT_RAIN + NE_PLATEAU_WIDTH * VERY_HIGH_RAIN * NE_PLATEAU_HEIGHT
pub const NE_PLATEAU_WIDTH: usize = ((MAIN_CENTRAL_HIGHLAND_RAIN - NORTH_DESERT_RAIN) / (VERY_HIGH_RAIN * NE_PLATEAU_HEIGHT as f32 - LOW_RAIN * CENTRAL_HIGHLAND_HEIGHT as f32)) as usize;
pub const TOTAL_LAND_WIDTH: usize = NE_PLATEAU_WIDTH + NORTH_DESERT_WIDTH;
pub const CONTINENTAL_SHELF_WIDTH: usize = (50.0 / HEX_FACTOR) as usize;
pub const CONTINENTAL_SLOPE_WIDTH: usize = WIDTH_HEXAGONS - TOTAL_LAND_WIDTH - CONTINENTAL_SHELF_WIDTH;
pub const CONTINENTAL_SHELF_DEPTH: f32 = 460.0;
pub const CONTINENTAL_SLOPE_INCREMENT: f32 = (16_762.0 - CONTINENTAL_SHELF_DEPTH) / CONTINENTAL_SLOPE_WIDTH as f32;
pub const SEA_LEVEL: f32 = (CONTINENTAL_SLOPE_WIDTH as f32) * CONTINENTAL_SLOPE_INCREMENT + CONTINENTAL_SHELF_DEPTH;
pub const TOTAL_SEA_WIDTH: usize = WIDTH_HEXAGONS - TOTAL_LAND_WIDTH;
pub const NORTH_DESERT_MAX_ELEVATION: f32 = 7_175.0;
pub const NORTH_DESERT_INCREMENT: f32 = (NORTH_DESERT_MAX_ELEVATION - RANDOM_ELEVATION_FACTOR) / (NORTH_DESERT_WIDTH - NE_PLATEAU_FRINGE) as f32;
pub const NE_PLATEAU_MAX_ELEVATION: f32 = 14_872.0;
pub const CENTRAL_HIGHLAND_MAX_ELEVATION: f32 = 10_131.0;
pub const CENTRAL_HIGHLAND_INCREMENT: f32 = (CENTRAL_HIGHLAND_MAX_ELEVATION - RANDOM_ELEVATION_FACTOR) / TOTAL_LAND_WIDTH as f32;
pub const SE_MOUNTAINS_MAX_ELEVATION: f32 = 18_510.0;
pub const SE_MOUNTAINS_INCREMENT: f32 = (SE_MOUNTAINS_MAX_ELEVATION - RANDOM_ELEVATION_FACTOR) / TOTAL_LAND_WIDTH as f32;
pub const SW_RANGE_MAX_ELEVATION: f32 = 16_854.0;
pub const SW_RANGE_FRINGE: usize = 4;
pub const SW_RANGE_HEIGHT: usize = 80;
pub const SW_RANGE_WIDTH: usize = (1080.0 / HEX_FACTOR) as usize;
// Unlike most other features, the SW range will be centered on the border between the central highland and the south mountains,
// with only a small fringe on either side.

pub const KC: f32 = 1.0; // capacity coefficient
pub const KE: f32 = 1.0 / 7.0; // erosion rate fraction
pub const KD: f32 = 1.0 / 7.0; // deposition rate fraction

// TODO: Remove constants associated with the hard-coded river source after fully replacing it with the NE plateau.
// 0.01 hours of average flow through the Aswan Dam.
pub const RIVER_WATER_PER_STEP: f32 = 0.371; // Feet
// In the river bed.
pub const TARGET_DROP_PER_HEX: f32 = 0.4; // Feet
pub const TARGET_RIVER_DEPTH: f32 = 32.0; // Feet
pub const FLOW_FACTOR: f32 = 0.9;
// One inch of rain per year.
pub const BASE_RAINFALL: f32 = 1.0 / 12.0 / 365.0 / 24.0 / 100.0; // Feet
pub const DEFAULT_ROUNDS: u32 = 1000;
pub const WATER_THRESHOLD: f32 = 1.0 / 12.0; // Feet


pub const MAX_SLOPE: f32 = 1.00;
pub const MAX_FLOW: f32 = (HEX_SIZE as f32) * MAX_SLOPE;
// Current highest of all max elevation constants.
pub const MAX_ELEVATION: f32 = SEA_LEVEL + SE_MOUNTAINS_MAX_ELEVATION;
pub const LOG_ROUNDS: u32 = 1;

pub const BIG_VOLCANO_INITIAL_ELEVATION: f32 = HEX_SIZE * (5.0 + 6.0 * 4.0 + 12.0 * 3.0 + 18.0 * 2.0 + 24.0 * 1.0);
pub const BIG_VOLCANO_X: usize = TOTAL_SEA_WIDTH + DELTA_SEED_WIDTH + (500.0 / HEX_FACTOR) as usize;