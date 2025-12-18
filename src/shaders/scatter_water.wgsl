struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _rainfall: f32,
    elevation_residual: f32,
    erosion_multiplier: f32,
    uplift: f32,
    water_depth_residual: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> next_water: array<atomic<i32>>;
@group(0) @binding(2)
var<storage, read> next_load: array<atomic<i32>>;

// Essentially a counterpart of init_water_step, maybe one or the other or both
// needs a better name.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if(index >= arrayLength(&hex_data)) { return; }

    // Get new water depth including existing residual
    let new_water_raw = max(bitcast<f32>(atomicLoad(&next_water[index])), 0.0);
    let total_water = new_water_raw + hex_data[index].water_depth_residual;

    // Truncate to 1/1024 precision (same as elevation) and store remainder in residual
    let adj = trunc(total_water * 1024.0) / 1024.0;
    hex_data[index].water_depth = adj;
    hex_data[index].water_depth_residual = total_water - adj;

    hex_data[index].suspended_load = max(bitcast<f32>(atomicLoad(&next_load[index])), 0.0);
} 