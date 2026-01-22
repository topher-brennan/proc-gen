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
// Use f32 for the base calculation, round to avoid truncation artifacts
pub const WIDTH_HEXAGONS_F: f32 = WIDTH_PIXELS as f32 / HEX_FACTOR;
pub const WIDTH_HEXAGONS: usize = (WIDTH_HEXAGONS_F + 0.5) as usize;

pub const DAYS_PER_YEAR: f64 = 365.2422;
pub const STEPS_PER_DAY: f64 = 24.0 * 60.0;
pub const YEARS_PER_STEP: f64 = 1.0 / DAYS_PER_YEAR / STEPS_PER_DAY;
// Above numbers are in inches per year, this can be adjusted to e.g. feet per year.
pub const MAX_EVAPORATION_PER_YEAR: f32 = 2.0;

// Top edge of the map is assumed to be 25 degrees south latitude.
pub const ONE_DEGREE_LATITUDE_MILES: f32 = 69.0;
// Use f32 for consistency with most calculations (avoid f32<->f64 conversions)
pub const TRANSITION_PERIOD: f32 = ONE_DEGREE_LATITUDE_MILES * 2.0;
pub const DEVIATION_PERIOD: f32 = 96.0;
// Keep float versions of Y coordinates to avoid stepped boundaries
pub const OUTLET_Y_F: f32 = 4.5 * ONE_DEGREE_LATITUDE_MILES * 2.0;
pub const OUTLET_Y: usize = OUTLET_Y_F as usize;
pub const SOURCE_Y_F: f32 = TRANSITION_PERIOD;
pub const SOURCE_Y: usize = SOURCE_Y_F as usize;
pub const MIN_NORTH_DESERT_HEIGHT_F: f32 = 6.5 * ONE_DEGREE_LATITUDE_MILES * 2.0;
pub const MIN_NORTH_DESERT_HEIGHT: usize = MIN_NORTH_DESERT_HEIGHT_F as usize;
pub const NE_BASIN_HEIGHT: usize = MIN_NORTH_DESERT_HEIGHT;
pub const NE_BASIN_HEIGHT_F: f32 = MIN_NORTH_DESERT_HEIGHT_F;
pub const CENTRAL_HIGHLAND_HEIGHT_F: f32 =
    11.5 * ONE_DEGREE_LATITUDE_MILES * 2.0 - MIN_NORTH_DESERT_HEIGHT_F;
pub const CENTRAL_HIGHLAND_HEIGHT: usize = CENTRAL_HIGHLAND_HEIGHT_F as usize;
pub const SOUTH_MOUNTAINS_HEIGHT_F: f32 =
    HEIGHT_PIXELS as f32 - MIN_NORTH_DESERT_HEIGHT_F - CENTRAL_HIGHLAND_HEIGHT_F;
pub const SOUTH_MOUNTAINS_HEIGHT: usize = SOUTH_MOUNTAINS_HEIGHT_F as usize;

pub const NORTH_DESERT_WIDTH_MILES: f32 = 800.0;
// Use float versions for calculations, integer versions for array indexing
pub const NORTH_DESERT_WIDTH_F: f32 = NORTH_DESERT_WIDTH_MILES * 2.0 / HEX_FACTOR;
pub const NORTH_DESERT_WIDTH: usize = (NORTH_DESERT_WIDTH_F + 0.5) as usize;
pub const COAST_WIDTH_F: f32 = 72.0 * 2.0 / HEX_FACTOR;
pub const COAST_WIDTH: usize = (COAST_WIDTH_F + 0.5) as usize;
pub const NE_BASIN_FRINGE: usize = 4;
pub const NE_BASIN_FRINGE_F: f32 = NE_BASIN_FRINGE as f32;
// An attempt to balance water in north and central regions mathematically.
// TODO: Maybe fix at 140?
pub const NE_BASIN_WIDTH_F: f32 = 100.0 * 2.0 / HEX_FACTOR;
pub const NE_BASIN_WIDTH: usize = (NE_BASIN_WIDTH_F + 0.5) as usize;
pub const NE_BASIN_RAIN: f32 =
    515_000.0 / NE_BASIN_HEIGHT_F / (NE_BASIN_WIDTH_F - NE_BASIN_FRINGE_F - 1.0);
pub const TOTAL_LAND_WIDTH_F: f32 = NE_BASIN_WIDTH_F + NORTH_DESERT_WIDTH_F;
pub const TOTAL_LAND_WIDTH: usize = NE_BASIN_WIDTH + NORTH_DESERT_WIDTH;

pub const ABYSSAL_PLAINS_MAX_DEPTH: f32 = -16_800.0;
pub const BASE_DEPTH_ADJUSTMENT: f32 = 0.85;
pub const LAKE_MIN_ELEVATION: f32 = 0.0;
pub const TOTAL_SEA_WIDTH_F: f32 = WIDTH_HEXAGONS_F - TOTAL_LAND_WIDTH_F;
pub const TOTAL_SEA_WIDTH: usize = WIDTH_HEXAGONS - TOTAL_LAND_WIDTH;
pub const NO_ISLANDS_ZONE_WIDTH_F: f32 = 500.0 * 2.0 / HEX_FACTOR;
pub const NO_ISLANDS_ZONE_WIDTH: usize = (NO_ISLANDS_ZONE_WIDTH_F + 0.5) as usize;
pub const ISLANDS_ZONE_WIDTH_F: f32 = TOTAL_SEA_WIDTH_F - NO_ISLANDS_ZONE_WIDTH_F;
pub const ISLANDS_ZONE_WIDTH: usize = TOTAL_SEA_WIDTH - NO_ISLANDS_ZONE_WIDTH;
pub const BASIN_X_BOUNDARY_F: f32 = TOTAL_SEA_WIDTH_F + NORTH_DESERT_WIDTH_F;
pub const BASIN_X_BOUNDARY: usize = TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH;
// TODO: Fix this—it's putting the river source outside the basin proper.
pub const RIVER_SOURCE_X: usize = BASIN_X_BOUNDARY + NE_BASIN_FRINGE - 1;
pub const SEA_LEVEL: f32 = 0.0;
pub const BASE_SEA_LEVEL: f32 = SEA_LEVEL;

pub const NORTH_DESERT_MAX_ELEVATION: f32 = 8_700.0;
pub const FAR_NORTH_DESERT_MAX_ELEVATION: f32 = 3_300.0;
pub const CENTRAL_HIGHLAND_MAX_ELEVATION: f32 = 10_100.0;
pub const SOUTH_MOUNTAINS_MAX_ELEVATION: f32 = 16_900.0;
pub const ISLANDS_MAX_ELEVATION: f32 = 11_200.0;
pub const OUTLET_ELEVATION: f32 = 200.0;
pub const BOUNDARY_ELEVATION: f32 = 2000.0;
pub const NE_BASIN_MIN_ELEVATION: f32 = NORTH_DESERT_WIDTH_MILES;
// pub const NE_BASIN_MIN_ELEVATION: f32 = BOUNDARY_ELEVATION;

pub const KC: f32 = 0.5; // capacity coefficient
pub const KE: f32 = 0.01; // erosion rate fraction
                          // Experimentally, a KD of 0.01 results in even filling of large lakes.
                          // Too high a value may result in water sloshing back and forth drilling
                          // pits in lakes, not sure where the limit is though.
pub const KD: f32 = 0.01; // deposition rate fraction

// Used to attempt to compensate for predictable loss of highest peaks over time.
pub const RAIN_BASED_UPLIFT_FACTOR: f32 = KC * KE;

pub const FLOW_FACTOR: f32 = 0.9;
// Might take 7k-10k rounds to carve out the river valley I want.
pub const DEFAULT_ROUNDS: u32 = 1_000;
// TODO: Might be nice to have two levels of water to display in generated images,
// a "hex-inch" could represent a river too deep to be forded but only a few tens of
// feet across (vs. about half a mile for the entire hex). But to mark sea that's
// readily navigable, you might want a threshold of five or six feet. One foot
// might work as a compromise (and show where relatively shallow-draft boats can
// move freely, even if ones with deeper draft couldn't).
pub const LOW_WATER_THRESHOLD: f32 = 1.0 / 12.0; // Feet
pub const HIGH_WATER_THRESHOLD: f32 = 6.0; // Feet

pub const SWAMP_THRESHOLD: f32 = -6.0;
pub const REEF_THRESHOLD: f32 = -200.0;

pub const MAX_SLOPE: f32 = 1.0;
pub const MAX_FLOW: f32 = (HEX_SIZE as f32) * MAX_SLOPE;
// Current highest of all max elevation constants.
pub const MAX_ELEVATION: f32 = SEA_LEVEL + SOUTH_MOUNTAINS_MAX_ELEVATION;
pub const LOG_ROUNDS: u32 = 1;
pub const SAVE_LOGS: u32 = 100;
pub const HEARTBEAT_ROUNDS: u32 = 1;
