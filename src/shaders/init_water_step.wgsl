// Initializes the "next" buffers for water routing from current hex_data values.
// This allows subsequent shaders to just modify these values rather than
// having to initialize them in multiple branches
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
var<storage, read> hex_data: array<Hex>;

@group(0) @binding(1)
var<storage, read_write> next_water: array<atomic<i32>>;

@group(0) @binding(2)
var<storage, read_write> next_load: array<atomic<i32>>;

@group(0) @binding(3)
var<storage, read_write> next_water_residual: array<atomic<i32>>;

@group(0) @binding(4)
var<storage, read_write> next_load_residual: array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&hex_data)) {
        return;
    }
    
    // Store as fixed-point integers for atomic operations
    // This allows water_routing to use atomicAdd instead of compare-exchange loops
    atomicStore(&next_water[index], to_fixed_point(hex_data[index].water_depth));
    atomicStore(&next_load[index], to_fixed_point(hex_data[index].suspended_load));
    atomicStore(&next_water_residual[index], to_fixed_point(hex_data[index].water_depth_residual));
    atomicStore(&next_load_residual[index], to_fixed_point(hex_data[index].suspended_load_residual));
}
