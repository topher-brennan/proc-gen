// Applies the ocean boundary condition to the west-most column (x=0).
// Sets water depth to sea level and flushes all suspended sediment.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _padding: f32,
};

struct Params {
    sea_level: f32,
    height: f32,
    width: f32, // Needed to calculate index
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.x;

    // Exit if this invocation is for a cell outside the map's height
    if (y >= u32(params.height)) {
        return;
    }

    // The index corresponds to the cell at (x=0, y)
    let index = y; // Since we dispatch one workgroup per row on the west edge

    var cell = hex_data[index];

    let target_depth = max(params.sea_level - cell.elevation, 0.0);

    // Any water/sediment above this level is considered to have flowed into the ocean
    cell.water_depth = target_depth;
    cell.suspended_load = 0.0;

    hex_data[index] = cell;
} 