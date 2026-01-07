// Gather-based water routing using precomputed flow targets.
// Each hex pulls water FROM neighbors whose flow_target points to it.
// This eliminates all atomics with minimal extra memory reads.
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

@group(0) @binding(0)
var<storage, read> hex_data: array<Hex>;

@group(0) @binding(1)
var<storage, read_write> next_hex_data: array<Hex>;

@group(0) @binding(2)
var<storage, read> flow_target: array<u32>;

@group(0) @binding(3)
var<storage, read> min_elev: array<f32>;

fn is_valid_coord(x: i32, y: i32) -> bool {
    return x >= 0 && x < i32(WIDTH) && y >= 0 && y < i32(HEIGHT);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(WIDTH) + x);
}

fn get_offset(k: u32, even: bool) -> vec2<i32> {
    if (even) {
        switch(k) {
            case 0u: { return vec2<i32>(1,0);}   // E
            case 1u: { return vec2<i32>(0,1);}   // S
            case 2u: { return vec2<i32>(-1,0);}  // W
            case 3u: { return vec2<i32>(0,-1);}  // N
            case 4u: { return vec2<i32>(-1,-1);} // NW
            case 5u: { return vec2<i32>(1,-1);}  // NE
            default: { return vec2<i32>(0,0); }
        }
    } else {
        switch(k) {
            case 0u: { return vec2<i32>(1,0);}   // E
            case 1u: { return vec2<i32>(0,1);}   // S
            case 2u: { return vec2<i32>(-1,0);}  // W
            case 3u: { return vec2<i32>(0,-1);}  // N
            case 4u: { return vec2<i32>(-1,1);}  // SW
            case 5u: { return vec2<i32>(1,1);}   // SE
            default: { return vec2<i32>(0,0); }
        }
    }
}

// Compute outflow from source hex to a destination with given height
fn compute_outflow(source: Hex, dest_height: f32) -> vec4<f32> {
    let f = total_fluid(source);
    if (f <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let source_h = height(source);
    let diff = source_h - dest_height;
    if (diff <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let move_f = calculate_flow(f, diff);
    if (move_f <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let sed_frac = sediment_fraction(source);
    var water_out = (1.0 - sed_frac) * move_f;
    var load_out = sed_frac * move_f;
    var water_residual_out = 0.0;
    var load_residual_out = 0.0;
    
    // Small values go to residual
    if (water_out < 1.0 / 512.0) {
        water_residual_out = water_out;
        water_out = 0.0;
    }
    if (load_out < 1.0 / 512.0) {
        load_residual_out = load_out;
        load_out = 0.0;
    }
    
    return vec4<f32>(water_out, load_out, water_residual_out, load_residual_out);
}

@compute @workgroup_size(256)
fn route_water(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_hexes = u32(WIDTH * HEIGHT);
    
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % u32(WIDTH));
    let y = i32(index / u32(WIDTH));
    let even = (x & 1) == 0;
    
    let my_hex = hex_data[index];
    let my_h = height(my_hex);
    
    // Start with current values
    var new_water = my_hex.water_depth;
    var new_load = my_hex.suspended_load;
    var new_water_residual = my_hex.water_depth_residual;
    var new_load_residual = my_hex.suspended_load_residual;
    
    // === GATHER: Check each neighbor to see if their flow_target is ME ===
    for (var k = 0u; k < 6u; k = k + 1u) {
        let off = get_offset(k, even);
        let nx = x + off.x;
        let ny = y + off.y;
        
        if (is_valid_coord(nx, ny)) {
            let n_idx = get_hex_index(nx, ny);
            
            // Does this neighbor flow to me?
            if (flow_target[n_idx] == index) {
                // Yes! Compute how much water it sends
                let neighbor_hex = hex_data[n_idx];
                let inflow = compute_outflow(neighbor_hex, my_h);
                
                new_water += inflow.x;
                new_load += inflow.y;
                new_water_residual += inflow.z;
                new_load_residual += inflow.w;
            }
        }
    }
    
    // === Compute my outflow to my target ===
    let my_target = flow_target[index];
    if (my_target != index) {
        // I flow to someone else
        let dest_height = min_elev[index];
        let outflow = compute_outflow(my_hex, dest_height);
        new_water -= outflow.x;
        new_load -= outflow.y;
        new_water_residual -= outflow.z;
        new_load_residual -= outflow.w;
    }
    
    // === Write results (no atomics - each thread writes only to its own cell) ===
    next_hex_data[index].elevation = my_hex.elevation;
    next_hex_data[index].elevation_residual = my_hex.elevation_residual;
    next_hex_data[index].water_depth = max(new_water, 0.0);
    next_hex_data[index].water_depth_residual = new_water_residual;
    next_hex_data[index].suspended_load = max(new_load, 0.0);
    next_hex_data[index].suspended_load_residual = new_load_residual;
    next_hex_data[index].rainfall = my_hex.rainfall;
    next_hex_data[index].erosion_multiplier = my_hex.erosion_multiplier;
    next_hex_data[index].uplift = my_hex.uplift;
}
