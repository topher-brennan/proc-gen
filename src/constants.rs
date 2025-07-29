pub const HEX_SIZE: f32 = 2640.0; // Feet
pub const KC: f32 = HEX_SIZE * 0.01; // capacity coefficient
pub const KE: f32 = 0.01;  // erosion rate fraction
pub const KD: f32 = 0.01;  // deposition rate fraction

// 0.01 hours of average flow through the Aswan Dam.
pub const RIVER_WATER_PER_STEP: f32 = 0.371; // Feet
// In the river bed.
pub const TARGET_DROP_PER_HEX: f32 = 0.4; // Feet
pub const TARGET_RIVER_DEPTH: f32 = 32.0; // Feet
pub const FLOW_FACTOR: f32 = RIVER_WATER_PER_STEP / TARGET_DROP_PER_HEX;
// One inch of rain per year.
pub const BASE_RAINFALL: f32 = 1.0 / 12.0 / 365.0 / 24.0 / 100.0; // Feet
pub const DEFAULT_ROUNDS: u32 = 1000;
pub const WATER_THRESHOLD: f32 = 1.0 / 12.0; // Feet
pub const HEX_FACTOR: f32 = 0.8660254037844386;
// Recommended to use a 9:16 aspect ratio. 720x1280 is popular, I like to use
// multiples of 216 and 384 because my monitor is 2160x3840.
pub const HEIGHT_PIXELS: u16 = 2160;
pub const WIDTH_PIXELS: u16 = 3840;
pub const WIDTH_HEXAGONS: u16 = (WIDTH_PIXELS as f32 / HEX_FACTOR) as u16;
pub const MAX_SLOPE: f32 = 0.05;
pub const MAX_FLOW: f32 = (HEX_SIZE as f32) * MAX_SLOPE;
pub const MAX_ELEVATION: f32 = (WIDTH_HEXAGONS as f32) * 6.0 + HEX_SIZE / 100.0;
pub const SEA_LEVEL: f32 = (WIDTH_HEXAGONS as f32) * 2.0;
pub const ONE_DEGREE_LATITUDE: f32 = 69.0 * 5280.0 / HEX_SIZE;