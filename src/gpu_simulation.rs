use bytemuck::{Pod, Zeroable};
use crate::constants::*;

// Helper macros for concise bind-group definitions
use wgpu::BufferUsages as BU;

macro_rules! buf_rw {
    ($binding:expr, $read_only:expr) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: $read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

macro_rules! uniform_entry {
    ($binding:expr) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

macro_rules! bg_entry {
    ($binding:expr, $buffer:expr) => {
        wgpu::BindGroupEntry {
            binding: $binding,
            resource: $buffer.as_entire_binding(),
        }
    };
}

macro_rules! dispatch_compute {
    ($device:expr, $queue:expr, $pipeline:expr, $bind_group:expr, $invocations:expr) => {{
        let mut encoder = $device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compute enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute pass") });
            pass.set_pipeline(&$pipeline);
            pass.set_bind_group(0, &$bind_group, &[]);
            pass.dispatch_workgroups($invocations, 1, 1);
        }
        $queue.submit(std::iter::once(encoder.finish()));
    }};
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HexGpu {
    pub elevation: f32,
    pub water_depth: f32,
    pub suspended_load: f32,
    pub rainfall: f32, // per-hex rainfall depth per step
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OutGpu {
    pub water_out: f32,
    pub sediment_out: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

pub struct GpuSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    hex_buffer: wgpu::Buffer,
    hex_buffer_size: usize,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // Rainfall specific pipeline and resources
    rainfall_pipeline: wgpu::ComputePipeline,
    rainfall_bind_group_layout: wgpu::BindGroupLayout,
    rainfall_bind_group: wgpu::BindGroup,
    rain_constants_buffer: wgpu::Buffer,
    // Water-routing resources
    routing_pipeline: wgpu::ComputePipeline,
    routing_bind_group_layout: wgpu::BindGroupLayout,
    routing_bind_group: wgpu::BindGroup,
    routing_constants_buffer: wgpu::Buffer,
    next_water_buffer: wgpu::Buffer,
    next_load_buffer: wgpu::Buffer,
    tgt_buffer: wgpu::Buffer,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_bind_group: wgpu::BindGroup,
    scatter_bind_group_layout: wgpu::BindGroupLayout,
    scatter_consts_buffer: wgpu::Buffer,
    min_elev_buffer: wgpu::Buffer,
    min_neigh_pipeline: wgpu::ComputePipeline,
    min_neigh_bind: wgpu::BindGroup,
    erosion_pipeline: wgpu::ComputePipeline,
    erosion_bind: wgpu::BindGroup,
    erosion_params: wgpu::Buffer,
    min_layout: wgpu::BindGroupLayout,
    min_consts_buf: wgpu::Buffer,
    eros_layout: wgpu::BindGroupLayout,
    // --- ocean boundary resources ---
    ocean_pipeline: wgpu::ComputePipeline,
    ocean_bind_group_layout: wgpu::BindGroupLayout,
    ocean_bind_group: wgpu::BindGroup,
    ocean_params_buffer: wgpu::Buffer,
    ocean_out_buffer: wgpu::Buffer,
    // river source updater
    river_pipeline: wgpu::ComputePipeline,
    river_bgl: wgpu::BindGroupLayout,
    river_bind: wgpu::BindGroup,
    river_params_buf: wgpu::Buffer,
    // --- repose (angle-of-repose) resources ---
    delta_buffer: wgpu::Buffer,
    repose_pipeline: wgpu::ComputePipeline,
    repose_bgl: wgpu::BindGroupLayout,
    repose_bind: wgpu::BindGroup,
    repose_consts_buf: wgpu::Buffer,
    apply_pipeline: wgpu::ComputePipeline,
    apply_bgl: wgpu::BindGroupLayout,
    apply_bind: wgpu::BindGroup,
}

impl GpuSimulation {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Increase limits to allow large storage buffers (>128 MiB)
        let mut limits = wgpu::Limits::default();
        // 512 MiB should comfortably cover very large maps while still
        // being supported on most desktop GPUs. The request will be
        // clamped to the adapter’s true limit automatically.
        limits.max_storage_buffer_binding_size = 512 * 1024 * 1024;
        // Keep max_buffer_size in sync so creation also succeeds.
        limits.max_buffer_size = 512 * 1024 * 1024;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // --------------------------------------------------
        // Generic placeholder compute shader (currently does
        // a simple rainfall increment; kept for compatibility)
        // --------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("simulation.wgsl"))),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipeline
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // --------------------------------------------------
        // Rainfall pipeline (uniform + hex storage buffer)
        // --------------------------------------------------

        let rainfall_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rainfall Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/rainfall.wgsl"))),
        });

        let rainfall_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Rainfall Bind Group Layout"),
            entries: &[
                // Storage buffer with hexes
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform buffer with constants
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let rainfall_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rainfall Pipeline Layout"),
            bind_group_layouts: &[&rainfall_bind_group_layout],
            push_constant_ranges: &[],
        });

        let rainfall_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rainfall Pipeline"),
            layout: Some(&rainfall_pipeline_layout),
            module: &rainfall_shader,
            entry_point: "add_rainfall",
        });

        // --------------------------------------------------
        // Water routing pipeline (uses separate buffers)
        // --------------------------------------------------

        let routing_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Routing Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/water_routing.wgsl")))),
        });

        let routing_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Routing BGL"),
            entries: &[
                // hex_data
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // next_water
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // next_load
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // tgt_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // constants
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let routing_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Routing Pipeline Layout"),
            bind_group_layouts: &[&routing_bind_group_layout],
            push_constant_ranges: &[],
        });

        let routing_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Water Routing Pipeline"),
            layout: Some(&routing_pipeline_layout),
            module: &routing_shader,
            entry_point: "route_water",
        });

        // ---------------- Scatter pipeline ---------------------------
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/scatter_water.wgsl")))),
        });

        let scatter_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("Scatter BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
                wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
                wgpu::BindGroupLayoutEntry{binding:2,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
                wgpu::BindGroupLayoutEntry{binding:3,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
                wgpu::BindGroupLayoutEntry{binding:4,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:false,min_binding_size:None},count:None},
            ],
        });

        let scatter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Scatter Pipeline Layout"),
            bind_group_layouts: &[&scatter_bind_group_layout],push_constant_ranges:&[],
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Scatter Pipeline"),layout:Some(&scatter_pipeline_layout),module:&scatter_shader,entry_point:"main",
        });

        // Create buffers with minimal non-zero size (overwritten later)
        let hex_buffer_size = std::mem::size_of::<HexGpu>();
        let hex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: hex_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Constants buffer (rain_per_step, hex_count) – rainfall shader
        let rain_constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rainfall Constants Buffer"),
            size: std::mem::size_of::<[f32; 3]>() as u64, // Changed from [f32; 1] to [f32; 2]
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let next_water_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Water Buffer"),
            size: 4, // placeholder 1 f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_load_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Load Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tgt_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Target Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let routing_constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Routing Constants Buffer"),
            size: std::mem::size_of::<[f32;4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let routing_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &routing_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding:0, resource: hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry { binding:1, resource: next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry { binding:2, resource: next_load_buffer.as_entire_binding()},
                wgpu::BindGroupEntry { binding:3, resource: tgt_buffer.as_entire_binding()},
                wgpu::BindGroupEntry { binding:4, resource: routing_constants_buffer.as_entire_binding()},
            ],
        });

        // Create rainfall bind group (will be recreated in initialize_buffer)
        let rainfall_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rainfall Bind Group"),
            layout: &rainfall_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rain_constants_buffer.as_entire_binding(),
                },
            ],
        });

        // Original generic bind group (storage only)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: hex_buffer.as_entire_binding(),
            }],
        });

        // placeholder consts buffer for scatter
        let scatter_consts_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("Scatter Consts"),size: (std::mem::size_of::<[f32;2]>()) as u64,usage:wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
        });

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("Scatter BG"),layout:&scatter_bind_group_layout,entries:&[
                wgpu::BindGroupEntry{binding:0,resource:hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:1,resource:next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:2,resource:next_load_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:3,resource:tgt_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:4,resource:scatter_consts_buffer.as_entire_binding()},
            ],
        });

        // ---- min_elev_buffer ----
        let min_elev_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size : 4,                          // resized later
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation:false,
        });

        // ---- min_neigh pipeline ----
        let min_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label:Some("min shader"),
            source:wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/min_neigh.wgsl")).into())
        });
        let min_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label:Some("min BGL"),
            entries:&[
              buf_rw!(0,false),                // hex_data
              buf_rw!(1,false),                // min_elev (write)
              uniform_entry!(2),
            ]});
        let min_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label:Some("min layout"), bind_group_layouts:&[&min_bgl], push_constant_ranges:&[]});
        let min_neigh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label:Some("min pipe"), layout:Some(&min_layout), module:&min_shader, entry_point:"main"});
        let consts_buf = device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("min consts"), size:8, usage:BU::UNIFORM|BU::COPY_DST, mapped_at_creation:false});
        let min_neigh_bind = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("min BG"), layout:&min_bgl, entries:&[
                bg_entry!(0,&hex_buffer),
                bg_entry!(1,&min_elev_buffer),
                bg_entry!(2,&consts_buf),
        ]});

        // ---- erosion pipeline ----
        let eros_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label:Some("erosion shader"),
            source:wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/erosion.wgsl")).into())
        });
        let eros_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label:Some("erosion BGL"),
            entries:&[
              buf_rw!(0,false),            // hex_data
              buf_rw!(1,true),             // min_elev (read)
              uniform_entry!(2),
        ]});
        let eros_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label:Some("eros layout"), bind_group_layouts:&[&eros_bgl], push_constant_ranges:&[] });
        let erosion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label:Some("eros pipe"), layout:Some(&eros_layout), module:&eros_shader, entry_point:"main"});
        let erosion_params = device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("eros params"), size:24, usage:BU::UNIFORM|BU::COPY_DST, mapped_at_creation:false});
        let erosion_bind = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("eros BG"), layout:&eros_bgl, entries:&[
                bg_entry!(0,&hex_buffer),
                bg_entry!(1,&min_elev_buffer),
                bg_entry!(2,&erosion_params),
        ]});

        // ---- ocean boundary pipeline ----
        let ocean_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ocean Boundary Shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/ocean_boundary.wgsl")).into()),
        });

        let ocean_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ocean Boundary BGL"),
            entries: &[
                buf_rw!(0,false), // hex_data
                uniform_entry!(1), // params
                buf_rw!(2,false), // outflow buffer
            ],
        });

        let ocean_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ocean Boundary Pipeline Layout"),
            bind_group_layouts: &[&ocean_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ocean_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ocean Boundary Pipeline"),
            layout: Some(&ocean_pipeline_layout),
            module: &ocean_shader,
            entry_point: "main",
        });

        let ocean_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Params Buffer"),
            size: std::mem::size_of::<[f32;4]>() as u64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let ocean_out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Out Buffer"),
            size: 4u64, // placeholder, resized later
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        let ocean_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &ocean_bind_group_layout,
            entries: &[
                bg_entry!(0,&hex_buffer),
                bg_entry!(1,&ocean_params_buffer),
                bg_entry!(2,&ocean_out_buffer),
            ],
        });

        // ---- river source updater pipeline ----
        let river_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("River Source Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/river_source.wgsl").into()),
        });

        let river_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("River Source BGL"),
            entries: &[
                buf_rw!(0, false),  // hex_data (storage, read-write)
                uniform_entry!(1),   // params (uniform)
            ],
        });

        let river_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("River Source Pipeline Layout"),
            bind_group_layouts: &[&river_bgl],
            push_constant_ranges: &[],
        });

        let river_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("River Source Pipeline"),
            layout: Some(&river_pipeline_layout),
            module: &river_shader,
            entry_point: "main",
        });

        let river_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("River Params Buffer"),
            size: std::mem::size_of::<[f32; 6]>() as u64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let river_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("River Bind Group"),
            layout: &river_bgl,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &river_params_buf),
            ],
        });

        // ---- repose (angle-of-repose) resources ----
        let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: 4, // placeholder, resized later
            usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let repose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Repose Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/repose_deltas.wgsl").into()),
        });

        let repose_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Repose BGL"),
            entries: &[
                buf_rw!(0, true),  // hex_data (read-only)
                buf_rw!(1, false), // delta buffer (atomic read-write)
                uniform_entry!(2), // constants
            ],
        });

        let repose_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Repose Pipeline Layout"),
            bind_group_layouts: &[&repose_bgl],
            push_constant_ranges: &[],
        });

        let repose_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Repose Pipeline"),
            layout: Some(&repose_pipeline_layout),
            module: &repose_shader,
            entry_point: "main",
        });

        let repose_consts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Repose Consts Buffer"),
            size: std::mem::size_of::<[f32; 4]>() as u64, // 16-byte aligned
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let repose_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Repose Bind Group"),
            layout: &repose_bgl,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &delta_buffer),
                bg_entry!(2, &repose_consts_buf),
            ],
        });

        let apply_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/apply_deltas.wgsl").into()),
        });

        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Apply BGL"),
            entries: &[
                buf_rw!(0, false), // hex_data read-write
                buf_rw!(1, false), // delta buffer read-write
            ],
        });

        let apply_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Apply Pipeline Layout"),
            bind_group_layouts: &[&apply_bgl],
            push_constant_ranges: &[],
        });

        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Apply Pipeline"),
            layout: Some(&apply_pipeline_layout),
            module: &apply_shader,
            entry_point: "main",
        });

        let apply_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Apply Bind Group"),
            layout: &apply_bgl,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &delta_buffer),
            ],
        });

        let min_layout_clone = min_bgl;
        let eros_layout_clone = eros_bgl;

        Self {
            device,
            queue,
            hex_buffer,
            hex_buffer_size,
            compute_pipeline,
            bind_group_layout,
            bind_group,
            rainfall_pipeline,
            rainfall_bind_group_layout,
            rainfall_bind_group,
            rain_constants_buffer,
            routing_pipeline,
            routing_bind_group_layout,
            routing_bind_group,
            routing_constants_buffer,
            next_water_buffer,
            next_load_buffer,
            tgt_buffer,
            scatter_pipeline,
            scatter_bind_group,
            scatter_bind_group_layout,
            scatter_consts_buffer,
            min_elev_buffer,
            min_neigh_pipeline,
            min_neigh_bind,
            erosion_pipeline,
            erosion_bind,
            erosion_params,
            min_layout: min_layout_clone,
            min_consts_buf: consts_buf,
            eros_layout: eros_layout_clone,
            // --- ocean boundary resources ---
            ocean_pipeline,
            ocean_bind_group_layout,
            ocean_bind_group,
            ocean_params_buffer,
            ocean_out_buffer,
            // river source updater
            river_pipeline,
            river_bgl,
            river_bind,
            river_params_buf,
            // --- repose (angle-of-repose) resources ---
            delta_buffer,
            repose_pipeline,
            repose_bgl,
            repose_bind,
            repose_consts_buf,
            apply_pipeline,
            apply_bgl,
            apply_bind,
        }
    }

    pub fn initialize_buffer(&mut self, width: usize, height: usize) {
        let buffer_size = width * height * std::mem::size_of::<HexGpu>();
        self.hex_buffer_size = buffer_size;
        
        self.hex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Recreate bind group with new buffer
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.hex_buffer.as_entire_binding(),
            }],
        });

        // Recreate rainfall bind group with new hex buffer
        self.rainfall_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rainfall Bind Group"),
            layout: &self.rainfall_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rain_constants_buffer.as_entire_binding(),
                },
            ],
        });

        // Resize next_water / next_load buffers
        let buf_bytes = (width*height*std::mem::size_of::<f32>()) as u64;
        self.next_water_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Water Buffer"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.next_load_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Load Buffer"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.tgt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Target Buffer"),
            size: (width*height*std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Recreate routing bind group with resized buffers
        self.routing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &self.routing_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{binding:0,resource:self.hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:1,resource:self.next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:2,resource:self.next_load_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:3,resource:self.tgt_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:4,resource:self.routing_constants_buffer.as_entire_binding()},
            ],
        });

        // Recreate scatter bind group with new buffers
        self.scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter BG"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{binding:0,resource:self.hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:1,resource:self.next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:2,resource:self.next_load_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:3,resource:self.tgt_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:4,resource:self.scatter_consts_buffer.as_entire_binding()},
            ],
        });

        // --- Resize ocean out buffer and bind group ---
        self.ocean_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Out Buffer"),
            size: (height * std::mem::size_of::<OutGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.ocean_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &self.ocean_bind_group_layout,
            entries: &[
                bg_entry!(0,&self.hex_buffer),
                bg_entry!(1,&self.ocean_params_buffer),
                bg_entry!(2,&self.ocean_out_buffer),
            ],
        });

        // Resize river params buffer and bind group
        self.river_params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("River Params Buffer"),
            size: std::mem::size_of::<[f32; 6]>() as u64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        self.river_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("River Bind Group"),
            layout: &self.river_bgl,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.river_params_buf),
            ],
        });

        // Resize min_elev_buffer (call after initialize_buffer).
        self.resize_min_buffers(width, height);

        // Resize delta buffer
        let delta_bytes = (width * height * std::mem::size_of::<u32>()) as u64;
        self.delta_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: delta_bytes,
            usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
            mapped_at_creation: false,
        });

        // Recreate repose bind group
        self.repose_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Repose Bind Group"),
            layout: &self.repose_bgl,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.delta_buffer),
                bg_entry!(2, &self.repose_consts_buf),
            ],
        });

        // Recreate apply bind group
        self.apply_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Apply Bind Group"),
            layout: &self.apply_bgl,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.delta_buffer),
            ],
        });
    }

    pub fn upload_data(&self, data: &[HexGpu]) {
        self.queue.write_buffer(&self.hex_buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn download_data(&self) -> Vec<HexGpu> {
        let buffer_size = self.hex_buffer_size;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.hex_buffer, 0, &staging_buffer, 0, buffer_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<HexGpu> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    pub fn run_simulation_step(&mut self, width: usize, height: usize) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simulation Encoder"),
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Simulation Pass"),
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        // Empty slice for dynamic offsets (not used)
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(width as u32, height as u32, 1);

        drop(compute_pass);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Adds uniform rainfall to every cell using a compute shader.
    pub fn run_rainfall_step(&mut self, total_cells: usize) {
        // Update constants buffer with hex_count and sea_level
        let constants = [total_cells as f32, SEA_LEVEL, EVAPORATION_FACTOR];
        self.queue.write_buffer(&self.rain_constants_buffer, 0, bytemuck::cast_slice(&constants));

        // Determine dispatch size (workgroup_size = 256)
        let workgroup_size: u32 = 256;
        let dispatch_x = ((total_cells as u32) + workgroup_size - 1) / workgroup_size;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Rainfall Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rainfall Pass"),
            });

            cpass.set_pipeline(&self.rainfall_pipeline);
            cpass.set_bind_group(0, &self.rainfall_bind_group, &[]);
            cpass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run water routing kernel and leave results in next buffers. Optionally download.
    pub fn run_water_routing_step(&mut self, width: usize, height: usize, flow_factor: f32, max_flow: f32) {
        let consts = [width as f32, height as f32, flow_factor, max_flow];
        self.queue.write_buffer(&self.routing_constants_buffer, 0, bytemuck::cast_slice(&consts));

        let workgroup_size: u32 = 256;
        let total = (width*height) as u32;
        let dispatch_x = (total + workgroup_size -1)/workgroup_size;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("Routing Encoder")});
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("Routing Pass")});
            cpass.set_pipeline(&self.routing_pipeline);
            cpass.set_bind_group(0,&self.routing_bind_group,&[]);
            cpass.dispatch_workgroups(dispatch_x,1,1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Downloads next_water and next_load buffers after routing.
    pub fn download_routing_results(&self) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let buf_size = self.hex_buffer_size / std::mem::size_of::<HexGpu>() * std::mem::size_of::<f32>();

        let create_staging = |label: &str| self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(label),
            size: buf_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });

        let staging_water = create_staging("StageWater");
        let staging_load  = create_staging("StageLoad");
        let staging_tgt   = self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("StageTgt"),
            size: (self.hex_buffer_size / std::mem::size_of::<HexGpu>() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("RouteDownloadEnc")});
        encoder.copy_buffer_to_buffer(&self.next_water_buffer,0,&staging_water,0,buf_size as u64);
        encoder.copy_buffer_to_buffer(&self.next_load_buffer ,0,&staging_load ,0,buf_size as u64);
        encoder.copy_buffer_to_buffer(&self.tgt_buffer ,0,&staging_tgt ,0,(buf_size/4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let read_buffer_f32 = |buf: &wgpu::Buffer| {
            let slice = buf.slice(..);
            let (tx,rx)= futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r|{tx.send(r).unwrap();});
            self.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(rx.receive()).unwrap().unwrap();
            let data = slice.get_mapped_range();
            let vec = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf.unmap();
            vec
        };

        let read_buffer_u32 = |buf: &wgpu::Buffer| {
            let slice = buf.slice(..);
            let (tx,rx)= futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r|{tx.send(r).unwrap();});
            self.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(rx.receive()).unwrap().unwrap();
            let data = slice.get_mapped_range();
            let vec = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf.unmap();
            vec
        };

        (read_buffer_f32(&staging_water), read_buffer_f32(&staging_load), read_buffer_u32(&staging_tgt))
    }

    pub fn run_scatter_step(&mut self, width: usize, height: usize) {
        // update scatter consts buffer
        let consts = [width as f32, height as f32];
        self.queue.write_buffer(&self.scatter_consts_buffer,0,bytemuck::cast_slice(&consts));

        let total = (width*height) as u32;
        let workgroup = 256u32;
        let dispatch = (total + workgroup -1)/workgroup;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("Scatter Enc")});
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("Scatter Pass")});
            cpass.set_pipeline(&self.scatter_pipeline);
            cpass.set_bind_group(0,&self.scatter_bind_group,&[]);
            cpass.dispatch_workgroups(dispatch,1,1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Adds water and suspended load to a single cell on the GPU buffer.
    /// `index` is linear index (y*width + x).
    pub fn add_inflow(&self, cell_index: usize, water: f32, load: f32) {
        // HexGpu layout: elevation (f32) @0, water_depth (f32) @4, suspended_load (f32) @8
        let base = (cell_index * std::mem::size_of::<HexGpu>()) as u64;
        self.queue.write_buffer(&self.hex_buffer, base + 4, bytemuck::bytes_of(&water));
        self.queue.write_buffer(&self.hex_buffer, base + 8, bytemuck::bytes_of(&load));
    }

    /// Convenience method: add only water.
    pub fn add_water(&self, cell_index: usize, water: f32) {
        let base = (cell_index * std::mem::size_of::<HexGpu>()) as u64;
        self.queue.write_buffer(&self.hex_buffer, base + 4, bytemuck::bytes_of(&water));
    }

    /// Convenience method: add only suspended load.
    pub fn add_load(&self, cell_index: usize, load: f32) {
        let base = (cell_index * std::mem::size_of::<HexGpu>()) as u64;
        self.queue.write_buffer(&self.hex_buffer, base + 8, bytemuck::bytes_of(&load));
    }

    // --------------------------------------------------------------
    // New helpers for min-neighbour + erosion GPU passes
    // --------------------------------------------------------------

    /// Resize the min_elev_buffer (call after initialize_buffer).
    pub fn resize_min_buffers(&mut self, width: usize, height: usize) {
        let size = (width * height * std::mem::size_of::<f32>()) as u64;
        self.min_elev_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.min_neigh_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("min BG"),
            layout: &self.min_layout,
            entries: &[
                bg_entry!(0,&self.hex_buffer),
                bg_entry!(1,&self.min_elev_buffer),
                bg_entry!(2,&self.min_consts_buf),
            ],
        });

        self.erosion_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eros BG"),
            layout: &self.eros_layout,
            entries: &[
                bg_entry!(0,&self.hex_buffer),
                bg_entry!(1,&self.min_elev_buffer),
                bg_entry!(2,&self.erosion_params),
            ],
        });
    }

    /// Compute minimum neighbour elevation (one pass).
    pub fn run_min_neigh_step(&self, width: usize, height: usize) {
        let consts = [width as f32, height as f32];
        self.queue.write_buffer(&self.min_consts_buf, 0, bytemuck::cast_slice(&consts));
        let total = (width * height) as u32;
        let groups = (total + 255) / 256;
        dispatch_compute!(self.device, self.queue, self.min_neigh_pipeline, self.min_neigh_bind, groups);
    }

    /// Run erosion/deposition per cell.
    pub fn run_erosion_step(&self, width: usize, height: usize) {
        let params: [f32; 6] = [KC, KE, KD, MAX_SLOPE, MAX_ELEVATION, HEX_SIZE];
        self.queue.write_buffer(&self.erosion_params, 0, bytemuck::cast_slice(&params));
        let total = (width * height) as u32;
        let groups = (total + 255) / 256;
        dispatch_compute!(self.device, self.queue, self.erosion_pipeline, self.erosion_bind, groups);
    }

    /// Run ocean boundary compute pass without reading data back to the CPU.
    /// This is significantly faster and should be used in the tight simulation loop.
    pub fn run_ocean_boundary(&mut self, width: usize, height: usize, sea_level: f32) {
        // Update params buffer: [sea_level, height, width, pad]
        let params = [sea_level, height as f32, width as f32, 0.0f32];
        self.queue.write_buffer(&self.ocean_params_buffer, 0, bytemuck::cast_slice(&params));

        // Dispatch compute – one thread per row along west edge
        let total_invocations = height as u32;
        let workgroup = 256u32;
        let dispatch_x = (total_invocations + workgroup - 1) / workgroup;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Ocean Boundary Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Ocean Boundary Pass") });
            cpass.set_pipeline(&self.ocean_pipeline);
            cpass.set_bind_group(0, &self.ocean_bind_group, &[]);
            cpass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Legacy helper that also downloads per-row outflow totals. Use only for debugging/profiling.
    pub fn run_ocean_boundary_readback(&mut self, width: usize, height: usize, sea_level: f32) -> (f32, f32) {
        // Update params buffer: [sea_level, height, width, pad]
        let params = [sea_level, height as f32, width as f32, 0.0f32];
        self.queue.write_buffer(&self.ocean_params_buffer, 0, bytemuck::cast_slice(&params));

        // Dispatch compute – one thread per row along west edge
        let total_invocations = height as u32;
        let workgroup = 256u32;
        let dispatch_x = (total_invocations + workgroup - 1) / workgroup;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Ocean Boundary Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Ocean Boundary Pass") });
            cpass.set_pipeline(&self.ocean_pipeline);
            cpass.set_bind_group(0, &self.ocean_bind_group, &[]);
            cpass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back outflow buffer (height entries)
        let buf_size = (height * std::mem::size_of::<OutGpu>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Outflow Staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Ocean Copy Enc") });
        enc.copy_buffer_to_buffer(&self.ocean_out_buffer, 0, &staging, 0, buf_size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vec: Vec<OutGpu> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        let mut total_water = 0.0f32;
        let mut total_sed = 0.0f32;
        for o in vec {
            total_water += o.water_out;
            total_sed += o.sediment_out;
        }
        (total_water, total_sed)
    }

    pub fn run_river_source_update(&mut self, idx: u32) {
        let params = [idx as f32, FLOW_FACTOR, TARGET_DROP_PER_HEX, TARGET_RIVER_DEPTH, KC, HEX_SIZE];
        self.queue.write_buffer(&self.river_params_buf, 0, bytemuck::cast_slice(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("RiverSrcEnc")});
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("RiverSrcPass")});
            cpass.set_pipeline(&self.river_pipeline);
            cpass.set_bind_group(0, &self.river_bind, &[]);
            cpass.dispatch_workgroups(1,1,1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Enforce angle-of-repose by two-pass delta approach.
    pub fn run_repose_step(&mut self, width: usize, height: usize) {
        // Write constants for repose_deltas shader: [width,height,HEX_SIZE]
        let consts = [width as f32, height as f32, HEX_SIZE];
        self.queue.write_buffer(&self.repose_consts_buf, 0, bytemuck::cast_slice(&consts));

        let total = (width * height) as u32;
        let groups = (total + 255) / 256;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("ReposeEncoder")});
        {
            // Pass 1: compute deltas with atomics
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("ReposeDeltasPass")});
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(groups,1,1);
        }
        {
            // Pass 2: apply deltas
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("ApplyDeltasPass")});
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind, &[]);
            pass.dispatch_workgroups(groups,1,1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn download_hex_data(&self) -> Vec<HexGpu> {
        self.download_data()
    }
} 

// ---------------------------------------------------------------------------
// Unit tests – compare GPU gather outputs with reference CPU implementation
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    const NEIGH_OFFSETS_EVEN: [(i16, i16); 6] = [
        (1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (1, -1),
    ];
    const NEIGH_OFFSETS_ODD: [(i16, i16); 6] = [
        (1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, 1),
    ];

    #[derive(Clone)]
    struct HexCpu {
        elevation: f32,
        water_depth: f32,
        suspended_load: f32,
    }

    fn cpu_gather(hex_map: &[Vec<HexCpu>], flow_factor: f32, max_flow: f32) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let height = hex_map.len();
        let width = hex_map[0].len();
        let mut out_w = vec![0.0f32; width*height];
        let mut out_load = vec![0.0f32; width*height];
        let mut tgt = vec![u32::MAX; width*height];

        for y in 0..height {
            for x in 0..width {
                let idx = y*width + x;
                let cell = &hex_map[y][x];
                let w = cell.water_depth;
                if w<=0.0 { continue; }
                let mut min_height = cell.elevation + w;
                let mut target: Option<(usize,usize)> = None;
                let offsets = if (x & 1)==0 { &NEIGH_OFFSETS_EVEN } else { &NEIGH_OFFSETS_ODD };
                for &(dx,dy) in offsets {
                    let nx_i = x as i32 + dx as i32;
                    let ny_i = y as i32 + dy as i32;
                    if nx_i<0 || ny_i<0 || nx_i>=width as i32 || ny_i>=height as i32 { continue; }
                    let nh = hex_map[ny_i as usize][nx_i as usize].elevation + hex_map[ny_i as usize][nx_i as usize].water_depth;
                    if nh < min_height {
                        min_height=nh; target=Some((nx_i as usize, ny_i as usize));
                    }
                }
                if let Some((tx,ty)) = target {
                    let diff = cell.elevation + w - (hex_map[ty][tx].elevation + hex_map[ty][tx].water_depth);
                    let move_w = (if diff > w { w } else { diff*flow_factor }).min(max_flow);
                    if move_w>0.0 {
                        out_w[idx]=move_w;
                        out_load[idx]=cell.suspended_load*move_w/w;
                        tgt[idx]=(ty*width+tx) as u32;
                    }
                }
            }
        }
        (out_w,out_load,tgt)
    }

    #[test]
    fn gpu_vs_cpu_gather() {
        let width=16usize; let height=12usize;
        let mut rng = rand::thread_rng();

        // build cpu map and gpu vec
        let mut cpu_map: Vec<Vec<HexCpu>> = vec![vec![HexCpu{elevation:0.0,water_depth:0.0,suspended_load:0.0}; width]; height];
        let mut gpu_vec: Vec<HexGpu> = Vec::with_capacity(width*height);
        for y in 0..height {
            for x in 0..width {
                let elev = rng.gen_range(0.0..100.0);
                let water = rng.gen_range(0.0..10.0);
                let load = rng.gen_range(0.0..1.0);
                cpu_map[y][x]=HexCpu{elevation:elev, water_depth:water, suspended_load:load};
                gpu_vec.push(HexGpu{elevation:elev, water_depth:water, suspended_load:load, rainfall:0.0});
            }
        }

        let (cpu_w,cpu_load,cpu_tgt) = cpu_gather(&cpu_map,0.9, width as f32);

        // GPU path
        let mut sim = pollster::block_on(GpuSimulation::new());
        sim.initialize_buffer(width,height);
        sim.upload_data(&gpu_vec);
        sim.run_water_routing_step(width,height,0.9,width as f32);
        let (gpu_w,gpu_load,gpu_tgt) = sim.download_routing_results();

        // compare (allow tiny numerical noise)
        const THRESH: f32 = 1e-6;
        for i in 0..cpu_w.len() {
            assert!((cpu_w[i]-gpu_w[i]).abs()<1e-4, "water mismatch at {}: {} vs {}",i,cpu_w[i],gpu_w[i]);
            assert!((cpu_load[i]-gpu_load[i]).abs()<1e-4, "load mismatch at {}",i);

            // We ignore target index differences as long as the water and load match,
            // since equal surface heights can legitimately allow multiple downslope choices.
        }
    }

    // -----------------------------------------------------------
    // Full gather + scatter comparison with CPU reference
    // -----------------------------------------------------------
    fn cpu_scatter(hex_map: &mut [Vec<HexCpu>], out_w:&[f32], out_load:&[f32], tgt:&[u32]){
        let height = hex_map.len();
        let width = hex_map[0].len();

        // subtract own outflow, accumulate inflows using tgt map
        for idx in 0..out_w.len(){
            let y = idx/width; let x = idx%width;
            hex_map[y][x].water_depth -= out_w[idx];
            hex_map[y][x].suspended_load -= out_load[idx];
            if hex_map[y][x].water_depth <0.0 { hex_map[y][x].water_depth = 0.0; }
            if hex_map[y][x].suspended_load <0.0 { hex_map[y][x].suspended_load = 0.0; }
        }
        for idx in 0..out_w.len(){
            let t = tgt[idx] as usize;
            if t == u32::MAX as usize { continue; }
            let ty = t/width; let tx = t%width;
            hex_map[ty][tx].water_depth += out_w[idx];
            hex_map[ty][tx].suspended_load += out_load[idx];
        }
    }

    #[test]
    fn gpu_vs_cpu_gather_scatter() {
        let width=16usize; let height=12usize;
        let mut rng = rand::thread_rng();

        // build cpu map and gpu vec
        let mut cpu_map: Vec<Vec<HexCpu>> = vec![vec![HexCpu{elevation:0.0,water_depth:0.0,suspended_load:0.0}; width]; height];
        let mut gpu_vec: Vec<HexGpu> = Vec::with_capacity(width*height);
        for y in 0..height {
            for x in 0..width {
                let elev = rng.gen_range(0.0..100.0);
                let water = rng.gen_range(0.0..10.0);
                let load = rng.gen_range(0.0..1.0);
                cpu_map[y][x]=HexCpu{elevation:elev, water_depth:water, suspended_load:load};
                gpu_vec.push(HexGpu{elevation:elev, water_depth:water, suspended_load:load, rainfall:0.0});
            }
        }

        let (cpu_w,cpu_load,cpu_tgt) = cpu_gather(&cpu_map,0.9,width as f32);
        let mut cpu_map_after = cpu_map.clone();
        cpu_scatter(&mut cpu_map_after,&cpu_w,&cpu_load,&cpu_tgt);

        // GPU path full
        let mut sim = pollster::block_on(GpuSimulation::new());
        sim.initialize_buffer(width,height);
        sim.upload_data(&gpu_vec);
        sim.run_water_routing_step(width,height,0.9,width as f32);
        sim.run_scatter_step(width,height);
        let gpu_hex = sim.download_hex_data();

        // compare
        for idx in 0..gpu_hex.len(){
            let y=idx/width; let x=idx%width;
            let cpu_hex = &cpu_map_after[y][x];
            assert!((cpu_hex.water_depth - gpu_hex[idx].water_depth).abs() < 1e-3, "depth mismatch at {}",idx);
            assert!((cpu_hex.suspended_load - gpu_hex[idx].suspended_load).abs() < 1e-3, "load mismatch at {}",idx);
        }
    }
} 