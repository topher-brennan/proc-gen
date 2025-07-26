struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_buffer: array<Hex>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // For now, just add a small amount of rainfall to each hex
    // This is a placeholder that we'll expand step by step
    hex_buffer[index].water_depth += 0.001;
} 