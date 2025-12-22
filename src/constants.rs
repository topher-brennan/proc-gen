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
pub const VERY_HIGH_RAIN: f32 = 53.0;
pub const NE_BASIN_RAIN: f32 = 30.0;
pub const DAYS_PER_YEAR: f32 = 365.2422;
pub const STEPS_PER_DAY: f32 = 24.0 * 6.0;
pub const YEARS_PER_STEP: f32 = 1.0 / DAYS_PER_YEAR / STEPS_PER_DAY;
// Above numbers are in inches per year, this can be adjusted to e.g. feet per year.
pub const RAINFALL_FACTOR: f32 = YEARS_PER_STEP / 12.0;
// Some dubiously realistic back-of-the-envelope math suggested maybe I should use 4/3 and a threshold of 18 feet.
// Or maybe 2.4 and a threshold of 10 feet?
pub const EVAPORATION_FACTOR: f32 = YEARS_PER_STEP * 0.2;
pub const MAX_EVAPORATION_PER_STEP: f32 = 2.0 * YEARS_PER_STEP;

pub const ONE_DEGREE_LATITUDE_MILES: f32 = 69.0;
pub const TRANSITION_PERIOD: f64 = ONE_DEGREE_LATITUDE_MILES as f64 * 2.0;
pub const DEVIATION_PERIOD: f64 = 96.0;
pub const RIVER_Y: usize = (4.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NORTH_DESERT_HEIGHT: usize = (6.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize;
pub const NE_BASIN_HEIGHT: usize = NORTH_DESERT_HEIGHT;
pub const CENTRAL_HIGHLAND_HEIGHT: usize =
    (11.5 * ONE_DEGREE_LATITUDE_MILES * 2.0) as usize - NORTH_DESERT_HEIGHT;
pub const SOUTH_MOUNTAINS_HEIGHT: usize =
    HEIGHT_PIXELS - NORTH_DESERT_HEIGHT - CENTRAL_HIGHLAND_HEIGHT;

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
pub const COAST_FRINGE: usize = 18;
pub const NE_BASIN_FRINGE: usize = 4;
pub const NORTH_DESERT_RAIN: f32 = (COAST_WIDTH as f32 * LOW_RAIN
    + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * VERY_LOW_RAIN)
    * NORTH_DESERT_HEIGHT as f32;
// Rain on the part of the central highland whose east-west extent corresponds to the north desert.
pub const MAIN_CENTRAL_HIGHLAND_RAIN: f32 = (COAST_WIDTH as f32 * MEDIUM_RAIN
    + (NORTH_DESERT_WIDTH - COAST_WIDTH) as f32 * LOW_RAIN)
    * CENTRAL_HIGHLAND_HEIGHT as f32;
// An attempt to balance water in north and central regions mathematically.
// TODO: Maybe fix at 140?
pub const NE_BASIN_WIDTH: usize = (100.0 * 2.0 / HEX_FACTOR) as usize;
pub const TOTAL_LAND_WIDTH: usize = NE_BASIN_WIDTH + NORTH_DESERT_WIDTH;

pub const ABYSSAL_PLAINS_MAX_DEPTH: f32 = -16_300.0;
pub const LAKE_MIN_ELEVATION: f32 = 0.0;
pub const TOTAL_SEA_WIDTH: usize = WIDTH_HEXAGONS - TOTAL_LAND_WIDTH;
pub const NO_ISLANDS_ZONE_WIDTH: usize = (500.0 * 2.0 / HEX_FACTOR) as usize;
pub const ISLANDS_ZONE_WIDTH: usize = TOTAL_SEA_WIDTH - NO_ISLANDS_ZONE_WIDTH;
pub const RIVER_SOURCE_X: usize = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH - NE_BASIN_FRINGE + 1;
pub const SEA_LEVEL: f32 = 0.0;
pub const BASE_SEA_LEVEL: f32 = SEA_LEVEL;
// TODO: Do we still need this? Maybe for rainfall logic.
// Could maybe replace this with a maximum depth of surface currents of 330 feet.
pub const CONTINENTAL_SHELF_DEPTH: f32 = 1000.0;

pub const NORTH_DESERT_MAX_ELEVATION: f32 = 8_700.0;
pub const FAR_NORTH_DESERT_MAX_ELEVATION: f32 = 3_300.0;
pub const CENTRAL_HIGHLAND_MAX_ELEVATION: f32 = 10_100.0;
pub const SOUTH_MOUNTAINS_MAX_ELEVATION: f32 = 15_900.0;
pub const ISLANDS_MAX_ELEVATION: f32 = 11_200.0;
pub const OUTLET_ELEVATION: f32 = 200.0;
pub const BOUNDARY_ELEVATION: f32 = 2000.0;
pub const NE_BASIN_MIN_ELEVATION: f32 = 600.0;

pub const KC: f32 = 1.5; // capacity coefficient
pub const KE: f32 = 1.0 / 7.0; // erosion rate fraction
                               // Experimentally, a KD of 0.01 results in even filling of large lakes.
                               // Too high a value may result in water sloshing back and forth drilling
                               // pits in lakes, not sure where the limit is though.
pub const KD: f32 = 1.0 / 100.0; // deposition rate fraction

// Used to attempt to compensate for predictable loss of highest peaks over time (~1 foot per 30 years).
// Disabled as possible source of NaN values.
// pub const RAIN_BASED_UPLIFT_FACTOR: f32 = 1.0 * RAINFALL_FACTOR * KC * KE; // Feet per step
pub const RAIN_BASED_UPLIFT_FACTOR: f32 = 0.0;

pub const FLOW_FACTOR: f32 = 0.9;
// Might take 7k-10k rounds to carve out the river valley I want.
pub const DEFAULT_ROUNDS: u32 = 4_800;
// TODO: Might be nice to have two levels of water to display in generated images,
// a "hex-inch" could represent a river too deep to be forded but only a few tens of
// feet across (vs. about half a mile for the entire hex). But to mark sea that's
// readily navigable, you might want a threshold of five or six feet. One foot
// might work as a compromise (and show where relatively shallow-draft boats can
// move freely, even if ones with deeper draft couldn't).
pub const LOW_WATER_THRESHOLD: f32 = 1.0 / 12.0; // Feet
pub const HIGH_WATER_THRESHOLD: f32 = 6.0; // Feet

pub const MAX_SLOPE: f32 = 1.0;
pub const MAX_FLOW: f32 = (HEX_SIZE as f32) * MAX_SLOPE;
// Current highest of all max elevation constants.
pub const MAX_ELEVATION: f32 = SEA_LEVEL + SOUTH_MOUNTAINS_MAX_ELEVATION;
pub const LOG_ROUNDS: u32 = 100;

pub const BIG_VOLCANO_X: usize =
    TOTAL_SEA_WIDTH + DELTA_SEED_WIDTH + (1_000.0 / HEX_FACTOR) as usize;
