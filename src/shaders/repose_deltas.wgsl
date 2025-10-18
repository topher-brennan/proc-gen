// Calculates the change in elevation (delta) for each cell based on the angle of repose.
// If a cell is much higher than a neighbor, it "loses" elevation, and the neighbor "gains" it.
// This uses atomics to safely handle multiple neighbors trying to modify the same cell's delta.

struct Hex {
    elevation:          f32,
    water_depth:        f32,
    suspended_load:     f32,
    _rainfall:          f32,
    elevation_residual: f32,
    erosion_multiplier: f32,
};

struct Consts {
    width: f32,
    height: f32,
    hex_size: f32,
};

@group(0) @binding(0)
var<storage, read> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read_write> delta_buffer: array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> C: Consts;

// Standard hex grid utility functions
const EVEN_OFFSETS: array<vec2<i32>, 6> = array<vec2<i32>, 6>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,-1), vec2<i32>(1,-1));
const ODD_OFFSETS: array<vec2<i32>, 6> = array<vec2<i32>, 6>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,1), vec2<i32>(1,1));

fn is_valid_coord(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && x < i32(C.width) && y < i32(C.height);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(C.width) + x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_hexes = u32(C.width * C.height);
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % u32(C.width));
    let y = i32(index / u32(C.width));
    let elev = total_elevation(hex_data[index]);

    let even_col = (x & 1) == 0;

    for (var i = 0u; i < 6u; i = i + 1u) {
        var offset: vec2<i32>;
        switch(i) {
            case 0u: { offset = select(ODD_OFFSETS[0], EVEN_OFFSETS[0], even_col); }
            case 1u: { offset = select(ODD_OFFSETS[1], EVEN_OFFSETS[1], even_col); }
            case 2u: { offset = select(ODD_OFFSETS[2], EVEN_OFFSETS[2], even_col); }
            case 3u: { offset = select(ODD_OFFSETS[3], EVEN_OFFSETS[3], even_col); }
            case 4u: { offset = select(ODD_OFFSETS[4], EVEN_OFFSETS[4], even_col); }
            case 5u: { offset = select(ODD_OFFSETS[5], EVEN_OFFSETS[5], even_col); }
            default: {offset = vec2<i32>(0,0);} // should not happen
        }
        let nx = x + offset.x;
        let ny = y + offset.y;

        if (is_valid_coord(nx, ny)) {
            let neighbor_index = get_hex_index(nx, ny);
            let neighbor_elev = total_elevation(hex_data[neighbor_index]);
            let diff = elev - neighbor_elev;

            if (diff > C.hex_size) {
                // This slope is too steep. Move some elevation.
                let excess = (diff - C.hex_size) / 7.0;
                var old_val: u32;
                var new_val: u32;
                // TODO: Alternatives to busy waiting here?
                loop {
                    old_val = atomicLoad(&delta_buffer[index]);
                    let old_f32 = bitcast<f32>(old_val);
                    let new_f32 = old_f32 - excess;
                    new_val = bitcast<u32>(new_f32);
                    let result = atomicCompareExchangeWeak(&delta_buffer[index], old_val, new_val);
                    if (result.exchanged) {
                        break;
                    }
                }
                var neighbor_old_val: u32;
                var neighbor_new_val: u32;
                loop {
                    neighbor_old_val = atomicLoad(&delta_buffer[neighbor_index]);
                    let neighbor_old_f32 = bitcast<f32>(neighbor_old_val);
                    let neighbor_new_f32 = neighbor_old_f32 + excess;
                    neighbor_new_val = bitcast<u32>(neighbor_new_f32);
                    let result = atomicCompareExchangeWeak(&delta_buffer[neighbor_index], neighbor_old_val, neighbor_new_val);
                    if (result.exchanged) {
                        break;
                    }
                }
            }
        }
    }
} 