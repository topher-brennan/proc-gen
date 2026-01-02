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
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<storage, read_write> next_water: array<atomic<i32>>;

@group(0) @binding(2)
var<storage, read_write> next_load: array<atomic<i32>>;

@group(0) @binding(3)
var<storage, read_write> next_water_residual: array<atomic<i32>>;

@group(0) @binding(4)
var<storage, read_write> next_load_residual: array<atomic<i32>>;

const NO_TARGET: u32 = 0xFFFFFFFFu;

// Neighbor offsets for even columns
const NEIGH_OFFSETS_EVEN: array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>(1, 0),   // 4 o'clock (east)
    vec2<i32>(0, 1),   // 6 o'clock (south)
    vec2<i32>(-1, 0),  // 8 o'clock (west)
    vec2<i32>(0, -1),  // 12 o'clock (north)
    vec2<i32>(-1, -1), // 10 o'clock (north-west)
    vec2<i32>(1, -1),  // 2 o'clock (north-east)
);

// Neighbor offsets for odd columns
const NEIGH_OFFSETS_ODD: array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>(1, 0),   // 2 o'clock (east)
    vec2<i32>(0, 1),   // 6 o'clock (south)
    vec2<i32>(-1, 0),  // 10 o'clock (west)
    vec2<i32>(0, -1),  // 12 o'clock (north)
    vec2<i32>(-1, 1),  // 8 o'clock (south-west)
    vec2<i32>(1, 1),   // 4 o'clock (south-east)
);

fn get_neighbor_coord(x: i32, y: i32, offset: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(x + offset.x, y + offset.y);
}

fn is_valid_coord(x: i32, y: i32) -> bool {
    return x >= 0 && x < i32(WIDTH) && y >= 0 && y < i32(HEIGHT);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(WIDTH) + x);
}


@compute @workgroup_size(256)
fn route_water(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let width = u32(WIDTH);
    let height = u32(HEIGHT);
    let total_hexes = width * height;
    
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % width);
    let y = i32(index / width);
    
    let hex = hex_data[index];
    let f = total_fluid(hex);
    
    if (f <= 0.0) {
        return;
    }
    
    // Find lowest neighbor (variables must be mutable, use 'var')
    var min_height: f32 = height(hex);
    var target_x: i32 = x;
    var target_y: i32 = y;
    
    let even_col = (x & 1) == 0;
    for (var i = 0u; i < 6u; i = i + 1u) {
        var offset: vec2<i32> = vec2<i32>(0, 0);
        if (even_col) {
            switch(i) {
                case 0u: { offset = vec2<i32>(1, 0); }
                case 1u: { offset = vec2<i32>(0, 1); }
                case 2u: { offset = vec2<i32>(-1, 0); }
                case 3u: { offset = vec2<i32>(0, -1); }
                case 4u: { offset = vec2<i32>(-1, -1); }
                case 5u: { offset = vec2<i32>(1, -1); }
                default: {}
            }
        } else {
            switch(i) {
                case 0u: { offset = vec2<i32>(1, 0); }
                case 1u: { offset = vec2<i32>(0, 1); }
                case 2u: { offset = vec2<i32>(-1, 0); }
                case 3u: { offset = vec2<i32>(0, -1); }
                case 4u: { offset = vec2<i32>(-1, 1); }
                case 5u: { offset = vec2<i32>(1, 1); }
                default: {}
            }
        }
        let nx = x + offset.x;
        let ny = y + offset.y;
        
        if (is_valid_coord(nx, ny)) {
            let n_index = get_hex_index(nx, ny);
            let n_hex = hex_data[n_index];
            let nh = height(n_hex);
            
            if (nh < min_height) {
                min_height = nh;
                target_x = nx;
                target_y = ny;
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

            // Subtract outflow from our own next buffers (atomic for thread safety)
            // We use compareExchange loop because atomicAdd only works on integers
            var old_bits = atomicLoad(&next_water[index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 - water_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_water[index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }
            
            old_bits = atomicLoad(&next_load[index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 - load_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_load[index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }
            
            old_bits = atomicLoad(&next_water_residual[index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 - residual_water_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_water_residual[index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }

            old_bits = atomicLoad(&next_load_residual[index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 - residual_load_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_load_residual[index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }

            // Add inflow to target's next buffers (atomic for thread safety)
            old_bits = atomicLoad(&next_water[target_index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 + water_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_water[target_index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }
            
            old_bits = atomicLoad(&next_load[target_index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 + load_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_load[target_index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }
            
            old_bits = atomicLoad(&next_water_residual[target_index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 + residual_water_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_water_residual[target_index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }

            old_bits = atomicLoad(&next_load_residual[target_index]);
            loop {
                let old_f32 = bitcast<f32>(old_bits);
                let new_f32 = old_f32 + residual_load_outflow;
                let new_bits = bitcast<i32>(new_f32);
                let result = atomicCompareExchangeWeak(&next_load_residual[target_index], old_bits, new_bits);
                if (result.exchanged) { break; }
                old_bits = result.old_value;
            }
        }
    }
} 