// GPU implementation of erosion / deposition per cell.
// Matches logic in main_original.rs Phase-4.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> min_elev: array<f32>;   // output from min_neigh pass

struct Params {
    kc: f32,
    ke: f32,
    kd: f32,
    max_slope: f32,
    max_elev: f32,
    hex_size: f32,
};
@group(0) @binding(2)
var<uniform> P : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&hex_data)) { return; }

    var cell = hex_data[idx];

    // Slope and capacity
    let slope = max((height(cell) - min_elev[idx]) / P.hex_size, 0.0);
    let capacity = P.kc * cell.water_depth * min(slope, P.max_slope);

    if (cell.suspended_load < capacity) {
        // erode
        // TODO: Look for better ways to prevent sea floor from being eroded below 0?
        let amount = (capacity - cell.suspended_load);
        cell.elevation      -= amount;
        cell.suspended_load += amount;
    } else {
        // deposit
        var amount = P.kd * (cell.suspended_load - capacity);
        if (cell.elevation + amount > P.max_elev) {
            amount = P.max_elev - cell.elevation;
        }
        cell.elevation      += amount;
        cell.suspended_load -= amount;
    }

    hex_data[idx] = cell;
} 