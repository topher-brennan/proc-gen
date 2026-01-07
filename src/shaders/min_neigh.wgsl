// Computes the minimum neighbour elevation and flow target for each hex cell.
// Output: 
//   min_elev[index] = lowest neighbour height (including self)
//   flow_target[index] = index of lowest neighbor (or self if no lower neighbor)
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

@group(0) @binding(0)
var<storage, read> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read_write> min_elev: array<f32>;
@group(0) @binding(2)
var<storage, read_write> flow_target: array<u32>;

fn inside(x:i32, y:i32) -> bool {
    return x >= 0 && y >= 0 && x < i32(WIDTH) && y < i32(HEIGHT);
}

fn idx(x:i32, y:i32) -> u32 {
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    let total = u32(WIDTH * HEIGHT);
    if (index >= total) { return; }

    let x = i32(index % u32(WIDTH));
    let y = i32(index / u32(WIDTH));

    var m = height(hex_data[index]);
    var target_idx = index;  // Default: flow to self (no lower neighbor)
    let even = (x & 1) == 0;

    for (var k:u32 = 0u; k < 6u; k = k + 1u) {
        let off = get_offset(k, even);
        let nx = x + off.x;
        let ny = y + off.y;
        if (!inside(nx, ny)) { continue; }
        let n_idx = idx(nx, ny);
        let elev = height(hex_data[n_idx]);
        if (elev < m) {
            m = elev;
            target_idx = n_idx;
        }
    }

    min_elev[index] = m;
    flow_target[index] = target_idx;
}
