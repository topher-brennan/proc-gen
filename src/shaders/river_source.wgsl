struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

struct Params {
    // Linear index of the source hex, passed as float to keep the whole
    // parameter block 32-bit floats from the Rust side. Convert to u32 in shader.
    idx: f32,
    flow_factor: f32,
    target_drop: f32,
    target_depth: f32,
    kc: f32,
    hex_size: f32,
};

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
    let i: u32 = u32(params.idx);
    var cell = hex_data[i];

    let move_w = params.flow_factor * (params.target_drop + params.target_depth - cell.water_depth);
    let delta_w = clamp(move_w, 0.0, params.target_depth);

    cell.water_depth = cell.water_depth + delta_w;
    let load_add = params.flow_factor * params.target_drop / params.target_depth * params.kc * params.target_drop / params.hex_size;
    cell.suspended_load = cell.suspended_load + load_add;

    hex_data[i] = cell;
} 