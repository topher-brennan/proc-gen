// Applies the ocean boundary condition to the west-most column (x=0).
// Sets water depth to sea level and flushes all suspended sediment.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _padding: f32,
};

struct Outflow {
    water_out: f32,
    sediment_out: f32,
    _pad1: f32,
    _pad2: f32,
};

// Update Params to 4 floats for 16-byte alignment
struct Params {
    sea_level: f32,
    height: f32,
    width: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> params: Params;

// New storage buffer for per-row outflow values
@group(0) @binding(2)
var<storage, read_write> out_data: array<Outflow>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.x;

    // Exit if this invocation is for a cell outside the map's height
    if (y >= u32(params.height)) {
        return;
    }

    // Compute linear index for west-edge cell at row y: idx = y * width + 0
    let index = y * u32(params.width);

    var cell = hex_data[index];

    var water_out: f32 = 0.0;
    var sediment_out: f32 = 0.0;
    if (height(cell) > params.sea_level) {
        let fluid_out: f32 = height(cell) - params.sea_level;
        water_out = fluid_out * (1.0 - sediment_fraction(cell));
        // Experiment to see if ocean sediment outflows could be causing runaway errosion.
        // sediment_out = fluid_out * sediment_fraction(cell);
    }

    // Write outflow amounts
    out_data[index].water_out = water_out;
    out_data[index].sediment_out = sediment_out;

    // Apply ocean boundary
    cell.water_depth -= water_out;
    cell.suspended_load -= sediment_out;

    hex_data[index] = cell;
} 