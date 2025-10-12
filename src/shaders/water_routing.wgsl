struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    rainfall: f32,
    elevation_residual: f32,
}

// TODO: read_write vs. read
@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;

@group(0) @binding(1)
var<storage, read_write> next_water: array<f32>;

@group(0) @binding(2)
var<storage, read_write> next_load: array<f32>;

@group(0) @binding(6)
var<storage, read_write> outflow_water: array<f32>;

@group(0) @binding(3)
var<storage, read_write> tgt_buffer: array<vec4<u32>>;

@group(0) @binding(5)
var<storage, read_write> flow_fractions: array<vec4<f32>>;

struct Constants {
    width: f32,
    height: f32,
    flow_factor: f32,
    max_flow: f32,
}

@group(0) @binding(4)
var<uniform> constants: Constants;

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
    return x >= 0 && x < i32(constants.width) && y >= 0 && y < i32(constants.height);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(constants.width) + x);
}

@compute @workgroup_size(256)
fn route_water(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let width = u32(constants.width);
    let height = u32(constants.height);
    let total_hexes = width * height;
    
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % width);
    let y = i32(index / width);
    
    let hex = hex_data[index];
    let f = total_fluid(hex);
    
    if (f <= 0.0) {
        next_water[index] = hex.water_depth;
        next_load[index] = 0.0;
        outflow_water[index] = 0.0;
        tgt_buffer[index]  = vec4<u32>(NO_TARGET, NO_TARGET, NO_TARGET, NO_TARGET);
        flow_fractions[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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
        // Old formula based on two-way fork rather than three-way
        // let move_f = min(select(diff * constants.flow_factor, f, diff > f), constants.max_flow);
        var move_f = 0.0;
        if (2.0 * f <= diff) {
            move_f = f;
        } else if (f < diff && diff < 2.0 * f) {
            move_f = (diff - f) + (2.0 * f - diff) * constants.flow_factor;
        } else { // diff <= f
            move_f = diff * constants.flow_factor;
        }

        move_f = min(move_f, constants.max_flow);
        // Extra safety: never route more fluid than present
        move_f = min(move_f, f);

        if (move_f > 0.0) {
            let water_outflow = (1.0 - sediment_fraction(hex)) * move_f;
            let load_outflow = sediment_fraction(hex) * move_f;

            // Store water outflow for scatter phase
            outflow_water[index] = water_outflow;

            // Store remaining water after outflow (new behavior)
            next_water[index] = hex.water_depth - water_outflow;

            // Store sediment outflow as delta (old behavior)
            next_load[index] = load_outflow;

            tgt_buffer[index].x = target_index;
            flow_fractions[index].x = 1.0;  // 100% flow to first target
        } else {
            outflow_water[index] = 0.0;
            next_water[index] = hex.water_depth;
            next_load[index] = 0.0;
            tgt_buffer[index]  = vec4<u32>(NO_TARGET, NO_TARGET, NO_TARGET, NO_TARGET);
            flow_fractions[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    } else {
        outflow_water[index] = 0.0;
        next_water[index] = hex.water_depth;
        next_load[index] = 0.0;
        tgt_buffer[index] = vec4<u32>(NO_TARGET, NO_TARGET, NO_TARGET, NO_TARGET);
        flow_fractions[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
} 