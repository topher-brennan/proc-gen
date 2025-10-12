struct Hex {
    elevation: f32,
    water_depth: f32,
    suspended_load: f32,
    _rainfall: f32,
    elevation_residual: f32,
}

@group(0) @binding(0)
var<storage, read_write> hex_data: array<Hex>;
@group(0) @binding(1)
var<storage, read> next_water: array<f32>;
@group(0) @binding(2)
var<storage, read> next_load: array<f32>;
@group(0) @binding(3)
var<storage, read> tgt_buffer: array<vec4<u32>>;
@group(0) @binding(5)
var<storage, read> flow_fractions: array<vec4<f32>>;
@group(0) @binding(6)
var<storage, read> outflow_water: array<f32>;
@group(0) @binding(7)
var<storage, read> outflow_load: array<f32>;

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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total = u32(consts.width*consts.height);
    if(index>=total){ return; }

    // Start with water and sediment after outflow (from routing phase, new behavior)
    var new_water = next_water[index];
    var new_load = next_load[index];

    // Add inflows from neighbours
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
        if(tgt_buffer[n_idx].x == index){
            new_water += outflow_water[n_idx] * flow_fractions[n_idx].x;
            new_load += outflow_load[n_idx] * flow_fractions[n_idx].x;
        }
    }

    hex_data[index].water_depth = new_water;
    hex_data[index].suspended_load = new_load;
} 