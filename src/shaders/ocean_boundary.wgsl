// Applies the ocean boundary condition to the west-most column (x=0).
// Sets water depth to sea level and flushes all suspended sediment.
// Constants are prepended via include_str! in Rust

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

struct Outflow {
    water_out: f32,
    sediment_out: f32,
    _pad1: f32,
    _pad2: f32,
};

// Runtime parameter - sea level can change over time
struct RuntimeParams {
    sea_level: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> params: RuntimeParams;

// Storage buffer for per-row outflow values
@group(0) @binding(2)
var<storage, read_write> out_data: array<Outflow>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let width = u32(WIDTH);
    let height = u32(HEIGHT);
    let total_hexes = width * height;

    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % width);
    let y = i32(index / width);

    if (x > 0 && y > 0 && y < (i32(height) - 1)) {
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
    if (fluid_out < 0.0 || total_elevation(cell) > ABYSSAL_PLAINS_MAX_DEPTH) {
        sediment_out = fluid_out * sediment_fraction(cell);
    }

    // TODO: Fix this to work with residual channel.
    // Write outflow amounts (one entry per row)
    out_data[y].water_out = water_out;
    out_data[y].sediment_out = sediment_out;

    // Apply ocean boundary
    cell.water_depth -= water_out;
    cell.water_depth = max(cell.water_depth, 0.0);

    cell.suspended_load -= sediment_out;
    cell.suspended_load = max(cell.suspended_load, 0.0);

    hex_data[index] = cell;
} 