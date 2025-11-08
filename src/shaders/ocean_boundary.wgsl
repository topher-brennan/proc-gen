// Applies the ocean boundary condition to the west-most column (x=0).
// Sets water depth to sea level and flushes all suspended sediment.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
    erosion_multiplier: f32,
    uplift: f32,
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
    abyssal_plain_depth: f32,
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
    let index = global_id.x;
    let width = u32(params.width);
    let height = u32(params.height);
    let total_hexes = width * height;

    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % width);
    let y = i32(index / width);

    // TODO: This is supposed to result in outflows on the western and southern edges of the
    // map. I could've sworn it was working before but now the southern outflows don't seem
    // to work.
    if (x != 0 && y != (i32(height) - 1)) {
        return;
    }

    var cell = hex_data[index];

    var water_out: f32 = 0.0;
    var sediment_out: f32 = 0.0;

    // Note that it says "out" but there's no guard against negative values, allowing inflows.
    // Outflows cannot exceed what's available, but there's no limit on inflows.
    let fluid_out: f32 = min(height(cell) - params.sea_level, total_fluid(cell));
    water_out = fluid_out * (1.0 - sediment_fraction(cell));
    // TODO: Continue experimenting with turning sediment outflows on/off.
    if (fluid_out < 0.0 || total_elevation(cell) > -1.0 * params.abyssal_plain_depth) {
        sediment_out = fluid_out * sediment_fraction(cell);
    }

    // Write outflow amounts (one entry per row)
    out_data[y].water_out = water_out;
    out_data[y].sediment_out = sediment_out;

    // Apply ocean boundary
    cell.water_depth -= water_out;
    if (cell.water_depth < 0.0) {
        cell.water_depth = 0.0;
    }

    cell.suspended_load -= sediment_out;
    if (cell.suspended_load < 0.0) {
        cell.suspended_load = 0.0;
    }

    hex_data[index] = cell;
} 