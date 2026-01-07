// Gather-based water routing - NO ATOMICS!
// Each hex pulls water FROM neighbors that would flow to it.
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
var<storage, read> flow_target: array<u32>;

@group(0) @binding(2)
var<storage, read_write> next_hex_data: array<Hex>;

const NO_TARGET: u32 = 0xFFFFFFFFu;

// Neighbor offsets for even columns
const EVEN: array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>(1, 0),   // E
    vec2<i32>(0, 1),   // S
    vec2<i32>(-1, 0),  // W
    vec2<i32>(0, -1),  // N
    vec2<i32>(-1, -1), // NW
    vec2<i32>(1, -1),  // NE
);

// Neighbor offsets for odd columns
const ODD: array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>(1, 0),   // E
    vec2<i32>(0, 1),   // S
    vec2<i32>(-1, 0),  // W
    vec2<i32>(0, -1),  // N
    vec2<i32>(-1, 1),  // SW
    vec2<i32>(1, 1),   // SE
);

fn is_valid_coord(x: i32, y: i32) -> bool {
    return x >= 0 && x < i32(WIDTH) && y >= 0 && y < i32(HEIGHT);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(WIDTH) + x);
}

// Compute what a source hex would send to its destination
fn compute_outflow(source: Hex, dest: Hex) -> vec4<f32> {
    let f = total_fluid(source);
    if (f <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let diff = height(source) - height(dest);
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
    let width = u32(WIDTH);
    let height_val = u32(HEIGHT);
    let total_hexes = width * height_val;
    
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % width);
    let y = i32(index / width);
    let even_col = (x & 1) == 0;
    
    let my_hex = hex_data[index];
    let my_dest = flow_target[index];
    
    // Start with current values
    var new_water = my_hex.water_depth;
    var new_load = my_hex.suspended_load;
    var new_water_residual = my_hex.water_depth_residual;
    var new_load_residual = my_hex.suspended_load_residual;
    
    // === GATHER: Check each neighbor to see if they flow TO me ===
    for (var k: u32 = 0u; k < 6u; k = k + 1u) {
        var off: vec2<i32>;
        if (even_col) {
            switch(k) {
                case 0u: { off = vec2<i32>(1, 0); }
                case 1u: { off = vec2<i32>(0, 1); }
                case 2u: { off = vec2<i32>(-1, 0); }
                case 3u: { off = vec2<i32>(0, -1); }
                case 4u: { off = vec2<i32>(-1, -1); }
                case 5u: { off = vec2<i32>(1, -1); }
                default: {}
            }
        } else {
            switch(k) {
                case 0u: { off = vec2<i32>(1, 0); }
                case 1u: { off = vec2<i32>(0, 1); }
                case 2u: { off = vec2<i32>(-1, 0); }
                case 3u: { off = vec2<i32>(0, -1); }
                case 4u: { off = vec2<i32>(-1, 1); }
                case 5u: { off = vec2<i32>(1, 1); }
                default: {}
            }
        }
        
        let nx = x + off.x;
        let ny = y + off.y;
        
        if (is_valid_coord(nx, ny)) {
            let neighbor_idx = get_hex_index(nx, ny);
            
            // Does this neighbor flow TO me?
            if (flow_target[neighbor_idx] == index) {
                // Yes! Compute what they would send
                let neighbor_hex = hex_data[neighbor_idx];
                let inflow = compute_outflow(neighbor_hex, my_hex);
                
                // Add to our totals
                new_water += inflow.x;
                new_load += inflow.y;
                new_water_residual += inflow.z;
                new_load_residual += inflow.w;
            }
        }
    }
    
    // Calculate flow
    if (target_x != x || target_y != y) {
        let target_index = get_hex_index(target_x, target_y);
        let target_hex = hex_data[target_index];
        let diff = height(hex) - height(target_hex);

        let move_f = calculate_flow(f, diff);

        if (move_f > 0.0) {
            var water_outflow = (1.0 - sediment_fraction(hex)) * move_f;
            var residual_water_outflow = 0.0;
            var load_outflow = sediment_fraction(hex) * move_f;
            var residual_load_outflow = 0.0;
            
            // Residual moves proportionally to how much water moves relative to total water
            if (water_outflow < 1.0 / 512.0) {
                residual_water_outflow = water_outflow;
                water_outflow = 0.0;
            }

            if (load_outflow < 1.0 / 512.0) {
                residual_load_outflow = load_outflow;
                load_outflow = 0.0;
            }

            // Convert to fixed-point and use atomicAdd (single operation, no spinning!)
            // This is MUCH faster than the previous atomicCompareExchangeWeak loops
            let water_outflow_fixed = to_fixed_point(water_outflow);
            let load_outflow_fixed = to_fixed_point(load_outflow);
            let residual_water_outflow_fixed = to_fixed_point(residual_water_outflow);
            let residual_load_outflow_fixed = to_fixed_point(residual_load_outflow);

            // Subtract outflow from our own next buffers
            atomicSub(&next_water[index], water_outflow_fixed);
            atomicSub(&next_load[index], load_outflow_fixed);
            atomicSub(&next_water_residual[index], residual_water_outflow_fixed);
            atomicSub(&next_load_residual[index], residual_load_outflow_fixed);

            // Add inflow to target's next buffers
            atomicAdd(&next_water[target_index], water_outflow_fixed);
            atomicAdd(&next_load[target_index], load_outflow_fixed);
            atomicAdd(&next_water_residual[target_index], residual_water_outflow_fixed);
            atomicAdd(&next_load_residual[target_index], residual_load_outflow_fixed);
        }
    }
}
