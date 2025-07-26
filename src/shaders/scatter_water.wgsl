struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read_write> out_water: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out_load: array<f32>;
@group(0) @binding(3)
var<storage, read> tgt_buffer: array<u32>;

struct Constants {
    width: f32,
    height: f32,
}
@group(0) @binding(4)
var<uniform> consts: Constants;

const NO_TARGET:u32 = 0xFFFFFFFFu;

// neighbour offsets same as gather shader
const NEIGH_EVEN: array<vec2<i32>,6> = array<vec2<i32>,6>(
    vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,-1), vec2<i32>(1,-1)
);
const NEIGH_ODD: array<vec2<i32>,6> = array<vec2<i32>,6>(
    vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1), vec2<i32>(-1,1), vec2<i32>(1,1)
);

fn valid(x:i32,y:i32)->bool{
    return x>=0 && y>=0 && x<i32(consts.width) && y<i32(consts.height);
}

fn idx(x:i32,y:i32)->u32 { return u32(y*i32(consts.width)+x); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = u32(consts.width*consts.height);
    if(index>=total){ return; }

    let w_out = out_water[index];
    let l_out = out_load[index];

    // Subtract own outflow
    var wd = hex_data[index].water_depth - w_out;
    var ld = hex_data[index].suspended_load - l_out;
    if (wd < 0.0) { wd = 0.0; }
    if (ld < 0.0) { ld = 0.0; }

    // add inflows from neighbours
    let w = u32(consts.width);
    let x = i32(index % w);
    let y = i32(index / w);
    let even = (x & 1) == 0;
    for(var i: u32 =0u; i<6u; i=i+1u){
        var off: vec2<i32> = vec2<i32>(0,0);
        if(even){
            switch(i){
                case 0u: { off = vec2<i32>(1,0);} 
                case 1u: { off = vec2<i32>(0,1);} 
                case 2u: { off = vec2<i32>(-1,0);} 
                case 3u: { off = vec2<i32>(0,-1);} 
                case 4u: { off = vec2<i32>(-1,-1);} 
                case 5u: { off = vec2<i32>(1,-1);} 
                default: {}
            }
        } else {
            switch(i){
                case 0u: { off = vec2<i32>(1,0);} 
                case 1u: { off = vec2<i32>(0,1);} 
                case 2u: { off = vec2<i32>(-1,0);} 
                case 3u: { off = vec2<i32>(0,-1);} 
                case 4u: { off = vec2<i32>(-1,1);} 
                case 5u: { off = vec2<i32>(1,1);} 
                default: {}
            }
        }
        let nx = x + off.x; let ny = y + off.y;
        if(!valid(nx,ny)){continue;}
        let n_idx = idx(nx,ny);
        if(tgt_buffer[n_idx]==index){
            wd += out_water[n_idx];
            ld += out_load[n_idx];
        }
    }

    hex_data[index].water_depth = wd;
    hex_data[index].suspended_load = ld;
} 