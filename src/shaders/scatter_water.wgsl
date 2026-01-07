// NOTE: common.wgsl is prepended via concat! in Rust (provides fixed-point functions)

struct Hex {
    elevation: f32,
    elevation_residual: f32,
    water_depth: f32,
    water_depth_residual: f32,
    suspended_load: f32,
    suspended_load_residual: f32,
    rainfall: f32,
    erosion_multiplier: f32,
    uplift: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> next_water: array<atomic<i32>>;
@group(0) @binding(2)
var<storage, read> next_load: array<atomic<i32>>;
@group(0) @binding(3)
var<storage, read> next_water_residual: array<atomic<i32>>;
@group(0) @binding(4)
var<storage, read> next_load_residual: array<atomic<i32>>;

// Essentially a counterpart of init_water_step, maybe one or the other or both
// needs a better name.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if(index >= arrayLength(&hex_data)) { return; }

    // Convert from fixed-point back to float
    hex_data[index].water_depth = max(from_fixed_point(atomicLoad(&next_water[index])), 0.0);
    hex_data[index].suspended_load = max(from_fixed_point(atomicLoad(&next_load[index])), 0.0);
    hex_data[index].water_depth_residual = from_fixed_point(atomicLoad(&next_water_residual[index]));
    hex_data[index].suspended_load_residual = from_fixed_point(atomicLoad(&next_load_residual[index]));
}
