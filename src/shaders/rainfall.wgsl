struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> constants: Constants;

struct Constants {
    hex_count: f32,
}

@compute @workgroup_size(256)
fn add_rainfall(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(constants.hex_count)) {
        return;
    }
    
    // Add per-cell rainfall value to water depth
    hex_data[index].water_depth += hex_data[index].rainfall;
} 