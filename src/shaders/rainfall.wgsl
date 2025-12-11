// Constants are prepended via include_str! in Rust

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
    erosion_multiplier: f32,
    uplift: f32,
}

// Runtime uniform - changes each frame
struct RuntimeParams {
    seasonal_rain_multiplier: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> params: RuntimeParams;

@compute @workgroup_size(256)
fn add_rainfall(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let hex_count = u32(WIDTH * HEIGHT);
    if (index >= hex_count) {
        return;
    }
    let cell = hex_data[index];
    
    // Note to self: stop toying with removing evaporation, it's important to avoid weirdness
    // when a big lake empties.
    // Don't evaporate in basins at edge of map.
    if (index % u32(WIDTH) <= u32(BASIN_X_BOUNDARY)) {
        // Once a body of water reaches 10' deep, let it fill until it overflows,
        // creating a connection to the sea. But don't evaporate once already below
        // sea level.
        let effective_depth = min(min(cell.water_depth, 10.0), max(height(cell), 0.0));
        hex_data[index].water_depth -= EVAPORATION_FACTOR * effective_depth;
        hex_data[index].water_depth = max(hex_data[index].water_depth, 0.0);
    }
    hex_data[index].water_depth += cell.rainfall * params.seasonal_rain_multiplier;
} 