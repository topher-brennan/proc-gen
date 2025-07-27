pub const HEX_SIZE: f32 = 2640.0; // Feet
pub const KC: f32 = HEX_SIZE / 100.0; // capacity coefficient
pub const KE: f32 = 0.0725;  // erosion rate fraction
pub const KD: f32 = 0.0725;  // deposition rate fraction

// 0.1 hours of average flow through the Aswan Dam.
pub const RIVER_WATER_PER_STEP: f32 = 0.371; // Feet
// In the river bed.
pub const TARGET_DROP_PER_HEX: f32 = 0.4; // Feet
pub const TARGET_RIVER_DEPTH: f32 = 32.0; // Feet
pub const FLOW_FACTOR: f32 = RIVER_WATER_PER_STEP / TARGET_DROP_PER_HEX;
pub const MAX_SLOPE: f32 = 1.0; // Prevents runaway erosion by capping slope used in capacity calc
// I *think* this will determine how much river bed elevation declines per hex when the system is in equilibrium.
// but need to test.
pub const RIVER_LOAD_FACTOR: f32 = 0.4;
// One inch of rain per year.
pub const RAIN_PER_STEP: f32 = 1.0 / 12.0 / 365.0 / 24.0 / 100.0; // Feet
pub const DEFAULT_ROUNDS: u32 = 1000;
pub const WATER_THRESHOLD: f32 = 1.0 / 12.0; // One inch in feet
pub const HEX_FACTOR: f32 = 0.8660254037844386; 
pub const WIDTH_HEXAGONS: u16 = (WIDTH_PIXELS as f32 / HEX_FACTOR) as u16;
pub const HEIGHT_PIXELS: u16 = 216;
pub const WIDTH_PIXELS: u16 = 384;
pub const MAX_FLOW: f32 = WIDTH_HEXAGONS as f32;
pub const MAX_ELEVATION: f32 = (WIDTH_HEXAGONS as f32) * 4.0 + HEX_SIZE / 100.0;
pub const SEA_LEVEL: f32 = (WIDTH_HEXAGONS as f32) * 2.0;