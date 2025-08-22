// GPU implementation of erosion / deposition per cell.
// Matches logic in main_original.rs Phase-4.

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
};

struct Log {
    eroded: f32,
    deposited: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> min_elev: array<f32>;   // output from min_neigh pass
@group(0) @binding(3)
var<storage, read_write> erosion_log: array<Log>;

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
    var residual = cell.elevation_residual;
    var log_entry: Log = Log(0.0, 0.0, 0.0, 0.0);

    // Slope and capacity
    let height_diff = max((height(cell) - min_elev[idx]), 0.0);
    let capacity = P.kc * min(cell.water_depth, min(height_diff, P.hex_size));

    if (cell.suspended_load < capacity) {
        // erode
        let amount = P.ke * (capacity - cell.suspended_load);

        residual -= amount;
        let adj = trunc(residual * 512.0) / 512.0;
        cell.elevation += adj;
        residual -= adj;

        cell.elevation_residual = residual;
        cell.suspended_load += amount;
        log_entry.eroded = amount;
    } else {
        // deposit
        var amount = P.kd * (cell.suspended_load - capacity);
        if (cell.elevation + amount > P.max_elev) {
            amount = P.max_elev - cell.elevation;
        }

        residual += amount;
        let adj = trunc(residual * 1024.0) / 1024.0;
        cell.elevation += adj;
        residual -= adj;

        cell.elevation_residual = residual;
        cell.suspended_load -= amount;
        log_entry.deposited = amount;
    }

    hex_data[idx] = cell;
    erosion_log[idx] = log_entry;
} 