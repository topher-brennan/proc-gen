// Applies the deltas calculated in the repose_deltas pass to the main hex_data buffer.

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

    let delta_bits = atomicLoad(&delta_buffer[index]);
    let delta_f32 = bitcast<f32>(delta_bits);

    if (delta_f32 != 0.0) {
        if (abs(delta_f32) < 512.0) {
            hex_data[index].elevation_residual = hex_data[index].elevation_residual + delta_f32;
        } else {
            hex_data[index].elevation = hex_data[index].elevation + delta_f32;
        }
    }

    // Reset delta buffer
    atomicStore(&delta_buffer[index], 0u);
}
