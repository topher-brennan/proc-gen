// Initialize next_water and next_load from hex_data at the start of each step
// This allows subsequent shaders to just modify these values rather than
// having to initialize them in multiple branches

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
}

@group(0) @binding(0)
var<storage, read> hex_data: array<Hex>;

@group(0) @binding(1)
var<storage, read_write> next_water: array<f32>;

@group(0) @binding(2)
var<storage, read_write> next_load: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&hex_data)) {
        return;
    }
    
    next_water[index] = hex_data[index].water_depth;
    next_load[index] = hex_data[index].suspended_load;
}

