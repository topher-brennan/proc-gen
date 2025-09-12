// Applies the deltas calculated in the repose_deltas pass to the main hex_data buffer.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read_write> delta_buffer: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&hex_data)) {
        return;
    }

    // Atomically load the f32 delta (stored as u32) and add it to the elevation.
    let delta_u32 = atomicLoad(&delta_buffer[index]);
    let delta_f32 = bitcast<f32>(delta_u32);

    if (delta_f32 != 0.0) {
        hex_data[index].elevation = hex_data[index].elevation + delta_f32;
    }

    // Reset the delta buffer for the next simulation step.
    atomicStore(&delta_buffer[index], 0u);
} 