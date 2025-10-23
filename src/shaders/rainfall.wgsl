struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
    erosion_multiplier: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> constants: Constants;

struct Constants {
    hex_count: f32,
    sea_level: f32,
    evaporation_factor: f32,
    width: f32,
    basin_x_boundary: f32,
    continental_shelf_depth: f32,
}

@compute @workgroup_size(256)
fn add_rainfall(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(constants.hex_count)) {
        return;
    }
    let cell = hex_data[index];
    
    // TODO: Consider gradually fading out rainfall rather than a hard stop.
    if (total_elevation(cell) >= constants.sea_level - constants.continental_shelf_depth) {
        // Don't evaporate in basins at edge of map.
        // if (index % u32(constants.width) <= u32(constants.basin_x_boundary)) {
            // Once a body of water reaches 18' deep, let it fill until it overflows,
            // creating a connection to the sea. But don't evaporate once already below
            // sea level.
            // let effective_depth = min(min(cell.water_depth, 18.0), max(height(cell), 0.0));
            // hex_data[index].water_depth -= constants.evaporation_factor * effective_depth;
        // }
        hex_data[index].water_depth += cell.rainfall;
    }
} 