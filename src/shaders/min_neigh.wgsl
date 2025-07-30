// Computes the minimum neighbour elevation for each hex cell.
// Output: min_elev[index] holds the lowest neighbour elevation (including self).

struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read_write> min_elev: array<f32>;

struct Consts {
    width:  f32,
    height: f32,
};
@group(0) @binding(2)
var<uniform> C : Consts;

// Neighbour offsets for even/odd columns (columns-line-up layout)
const EVEN : array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>( 1, 0), vec2<i32>( 0, 1), vec2<i32>(-1, 0),
    vec2<i32>( 0,-1), vec2<i32>(-1,-1), vec2<i32>( 1,-1)
);
const ODD  : array<vec2<i32>, 6> = array<vec2<i32>, 6>(
    vec2<i32>( 1, 0), vec2<i32>( 0, 1), vec2<i32>(-1, 0),
    vec2<i32>( 0,-1), vec2<i32>(-1, 1), vec2<i32>( 1, 1)
);

fn inside(x:i32, y:i32) -> bool {
    return x >= 0 && y >= 0 && x < i32(C.width) && y < i32(C.height);
}

fn idx(x:i32, y:i32) -> u32 {
    return u32(y * i32(C.width) + x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    let total = u32(C.width * C.height);
    if (index >= total) { return; }

    let x = i32(index % u32(C.width));
    let y = i32(index / u32(C.width));

    var m = height(hex_data[index]);
    let even = (x & 1) == 0;

    for (var k:u32 = 0u; k < 6u; k = k + 1u) {
        var off : vec2<i32> = vec2<i32>(0,0);
        if (even) {
            switch(k) {
                case 0u: { off = vec2<i32>(1,0);}  // E
                case 1u: { off = vec2<i32>(0,1);}  // S
                case 2u: { off = vec2<i32>(-1,0);} // W
                case 3u: { off = vec2<i32>(0,-1);} // N
                case 4u: { off = vec2<i32>(-1,-1);} // NW
                case 5u: { off = vec2<i32>(1,-1);}  // NE
                default: {}
            }
        } else {
            switch(k) {
                case 0u: { off = vec2<i32>(1,0);}  // E
                case 1u: { off = vec2<i32>(0,1);}  // S
                case 2u: { off = vec2<i32>(-1,0);} // W
                case 3u: { off = vec2<i32>(0,-1);} // N
                case 4u: { off = vec2<i32>(-1,1);}  // SW
                case 5u: { off = vec2<i32>(1,1);}   // SE
                default: {}
            }
        }
        let nx = x + off.x;
        let ny = y + off.y;
        if (!inside(nx, ny)) { continue; }
        let elev = height(hex_data[idx(nx, ny)]);
        if (elev < m) { m = elev; }
    }

    min_elev[index] = m;
} 