// Calculates the change in elevation (delta) for each cell based on the angle of repose.
// If a cell is much higher than a neighbor, it "loses" elevation, and the neighbor "gains" it.
// Uses atomicCompareExchange for f32 values stored as u32 bits.
// Constants and common.wgsl are prepended via include_str! in Rust

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
var<storage, read_write> delta_buffer: array<atomic<u32>>;

const EVEN_OFFSETS: array<vec2<i32>, 6> = array<vec2<i32>, 6>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,-1), vec2<i32>(1,-1));
const ODD_OFFSETS: array<vec2<i32>, 6> = array<vec2<i32>, 6>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,1), vec2<i32>(1,1));

fn is_valid_coord(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && x < i32(WIDTH) && y < i32(HEIGHT);
}

fn get_hex_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(WIDTH) + x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_hexes = u32(WIDTH * HEIGHT);
    if (index >= total_hexes) {
        return;
    }

    let x = i32(index % u32(WIDTH));
    let y = i32(index / u32(WIDTH));
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
            default: {offset = vec2<i32>(0,0);}
        }
        let nx = x + offset.x;
        let ny = y + offset.y;

        if (is_valid_coord(nx, ny)) {
            let neighbor_index = get_hex_index(nx, ny);
            let neighbor_elev = total_elevation(hex_data[neighbor_index]);
            let diff = elev - neighbor_elev;

            if (diff > HEX_SIZE) {
                let excess = (diff - HEX_SIZE) / 100.0;
                
                // Subtract from self (inlined atomic add f32 with -excess)
                {
                    var old_bits = atomicLoad(&delta_buffer[index]);
                    loop {
                        let old_f32 = bitcast<f32>(old_bits);
                        let new_f32 = old_f32 - excess;
                        let new_bits = bitcast<u32>(new_f32);
                        let result = atomicCompareExchangeWeak(&delta_buffer[index], old_bits, new_bits);
                        if (result.exchanged) { break; }
                        old_bits = result.old_value;
                    }
                }
                
                // Add to neighbor (inlined atomic add f32 with +excess)
                {
                    var old_bits = atomicLoad(&delta_buffer[neighbor_index]);
                    loop {
                        let old_f32 = bitcast<f32>(old_bits);
                        let new_f32 = old_f32 + excess;
                        let new_bits = bitcast<u32>(new_f32);
                        let result = atomicCompareExchangeWeak(&delta_buffer[neighbor_index], old_bits, new_bits);
                        if (result.exchanged) { break; }
                        old_bits = result.old_value;
                    }
                }
            }
        }
    }
}
