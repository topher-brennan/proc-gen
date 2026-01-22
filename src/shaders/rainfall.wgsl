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
}

// Runtime uniform - changes each frame
struct RuntimeParams {
    sea_level: f32,
    seasonal_rain_multiplier: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> min_elev: array<f32>;   // output from min_neigh pass
@group(0) @binding(2)
var<uniform> params: RuntimeParams;

@compute @workgroup_size(256)
fn add_rainfall(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let hex_count = u32(WIDTH * HEIGHT);
    if (index >= hex_count) {
        return;
    }
    var cell = hex_data[index];
    var water_residual = cell.water_depth_residual;
    var water = cell.water_depth;
    
    // Note to self: stop toying with removing evaporation, it's important to avoid weirdness
    // when a big lake empties.
    if (index % u32(WIDTH) <= u32(BASIN_X_BOUNDARY)) {
        // Realistically, evaporation should be proportional to the percentage of a hex
        // covered in water, but we don't actually track that. So we do a crude estimate
        // based on min_neigh and total water in the hex.
        // TODO: Maybe this should look at land height, not water surface? Would require some refactoring.
        let height_diff = max((height(cell) - min_elev[index]), 0.0);
        let total_water = total_water_depth(cell);
        var covered = 1.0;
        if (height_diff > 0.0) {
            covered = clamp(total_water / height_diff / 2.0, 0.0, 1.0);
        }

        // Regardless of min_neigh, need at least 3 feet of water to fully cover a hex.
        covered = clamp(covered, 0.0, total_water * total_water / 9.0);

        water_residual -= MAX_EVAPORATION_PER_YEAR * YEARS_PER_STEP * covered;

        // Add rainfall to residual
        water_residual += cell.rainfall * params.seasonal_rain_multiplier * YEARS_PER_STEP;
        
        if (water_residual + water < 0.0) {
            water_residual = 0.0;
            water = 0.0;
        }
    } else {
        // Basins don't have evaporation and their seasonal rainfall pattern is reversed.
        water_residual += cell.rainfall * (2.0 - params.seasonal_rain_multiplier) * YEARS_PER_STEP;
    }

    // Floor to 1/512 precision and apply
    //
    // If water_residual is negative, adj will be more negative, except maybe in case
    // of a rounding error, which the max() will prevent.
    let adj = trunc(water_residual * 512.0) / 512.0;
    hex_data[index].water_depth = water + adj;
    hex_data[index].water_depth_residual = water_residual - adj;
    // hex_data[index].water_depth = max(water, 0.0);
} 