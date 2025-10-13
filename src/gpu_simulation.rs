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
    pub residual_elevation: f32,
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
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // Rainfall specific pipeline and resources
    rainfall_pipeline: wgpu::ComputePipeline,
    rainfall_bind_group_layout: wgpu::BindGroupLayout,
    rainfall_bind_group: wgpu::BindGroup,
    rain_constants_buffer: wgpu::Buffer,
    // Water step initialization resources
    init_water_pipeline: wgpu::ComputePipeline,
    init_water_bind_group_layout: wgpu::BindGroupLayout,
    init_water_bind_group: wgpu::BindGroup,
    // Water-routing resources
    routing_pipeline: wgpu::ComputePipeline,
    routing_bind_group_layout: wgpu::BindGroupLayout,
    routing_bind_group: wgpu::BindGroup,
    routing_constants_buffer: wgpu::Buffer,
    next_water_buffer: wgpu::Buffer,
    next_load_buffer: wgpu::Buffer,
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
    // Ocean boundary resources
    ocean_pipeline: wgpu::ComputePipeline,
    ocean_bind_group_layout: wgpu::BindGroupLayout,
    ocean_bind_group: wgpu::BindGroup,
    ocean_params_buffer: wgpu::Buffer,
    ocean_out_buffer: wgpu::Buffer,
    erosion_log_buffer: wgpu::Buffer,
    // Repose (angle-of-repose) resources
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
                // TODO: experiment with just using default()?
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 512 * 1024 * 1024;
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

        // TODO: Remove?
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Rainfall pipeline
        let rainfall_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rainfall Shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/rainfall.wgsl")).into()),
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

        // Water step initialization pipeline (copies water/load from hex_data to next buffers)
        let init_water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Init Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/init_water_step.wgsl").into()),
        });

        let init_water_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Init Water BGL"),
            entries: &[
                // hex_data (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let init_water_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Init Water Pipeline Layout"),
            bind_group_layouts: &[&init_water_bind_group_layout],
            push_constant_ranges: &[],
        });

        let init_water_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init Water Pipeline"),
            layout: Some(&init_water_pipeline_layout),
            module: &init_water_shader,
            entry_point: "main",
        });

        // Water routing pipeline
        let routing_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Routing Shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/water_routing.wgsl")).into()),
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
                // next_water (atomic)
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
                // next_load (atomic)
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
                // constants
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // Scatter pipeline
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/scatter_water.wgsl")).into()),
        });

        let scatter_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("Scatter BGL"),
            entries: &[
                // hex_data (read_write)
                wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
                // next_water (atomic, read-only)
                wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
                // next_load (atomic, read-only)
                wgpu::BindGroupLayoutEntry{binding:2,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
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
            size: std::mem::size_of::<[f32; 6]>() as u64, // Uses 6 constants
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let next_water_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Water Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_load_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Load Buffer"),
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
                wgpu::BindGroupEntry { binding:3, resource: routing_constants_buffer.as_entire_binding()},
            ],
        });

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

        let init_water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Init Water Bind Group"),
            layout: &init_water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: next_water_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: next_load_buffer.as_entire_binding(),
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
            label:Some("Scatter Consts"),size: (std::mem::size_of::<[f32;6]>()) as u64,usage:wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
        });

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("Scatter BG"),layout:&scatter_bind_group_layout,entries:&[
                wgpu::BindGroupEntry{binding:0,resource:hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:1,resource:next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:2,resource:next_load_buffer.as_entire_binding()},
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
            label:Some("min consts"), size:std::mem::size_of::<[f32;2]>() as u64, usage:BU::UNIFORM|BU::COPY_DST, mapped_at_creation:false});
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
              buf_rw!(3,false),            // erosion log buffer
        ]});
        let eros_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label:Some("eros layout"), bind_group_layouts:&[&eros_bgl], push_constant_ranges:&[] });
        let erosion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label:Some("eros pipe"), layout:Some(&eros_layout), module:&eros_shader, entry_point:"main"});
        let erosion_params = device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("eros params"), size:24, usage:BU::UNIFORM|BU::COPY_DST, mapped_at_creation:false});
        // Create a placeholder erosion_log_buffer before binding (resized later)
        let erosion_log_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Erosion Log Buffer"),
            size: 4u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        let erosion_bind = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("eros BG"), layout:&eros_bgl, entries:&[
                bg_entry!(0,&hex_buffer),
                bg_entry!(1,&min_elev_buffer),
                bg_entry!(2,&erosion_params),
                bg_entry!(3,&erosion_log_buffer),
            ],
        });

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

        // erosion_log_buffer created above before bind group

        let ocean_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &ocean_bind_group_layout,
            entries: &[
                bg_entry!(0,&hex_buffer),
                bg_entry!(1,&ocean_params_buffer),
                bg_entry!(2,&ocean_out_buffer),
            ],
        });

        // repose (angle-of-repose) resources
        let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: 4, // placeholder, resized later
            usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let repose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Repose Shader"),
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/repose_deltas.wgsl")).into()),
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
            bind_group_layout,
            bind_group,
            rainfall_pipeline,
            rainfall_bind_group_layout,
            rainfall_bind_group,
            rain_constants_buffer,
            init_water_pipeline,
            init_water_bind_group_layout,
            init_water_bind_group,
            routing_pipeline,
            routing_bind_group_layout,
            routing_bind_group,
            routing_constants_buffer,
            next_water_buffer,
            next_load_buffer,
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
            erosion_log_buffer,
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

    /// Lightweight keep-alive to service the device without blocking.
    /// TODO: Not sure if this actually works.
    pub fn heartbeat(&self) {
        self.device.poll(wgpu::Maintain::Poll);
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

        // TODO: Re-examine some of this stuff with recreating bind groups and resizing stuff.
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

        // Recreate init_water bind group
        self.init_water_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Init Water Bind Group"),
            layout: &self.init_water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.next_water_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.next_load_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate routing bind group with resized buffers
        self.routing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &self.routing_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{binding:0,resource:self.hex_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:1,resource:self.next_water_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:2,resource:self.next_load_buffer.as_entire_binding()},
                wgpu::BindGroupEntry{binding:3,resource:self.routing_constants_buffer.as_entire_binding()},
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
            ],
        });

        // Resize ocean out buffer and bind group
        self.ocean_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Out Buffer"),
            size: (height * std::mem::size_of::<OutGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Resize erosion log buffer
        self.erosion_log_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Erosion Log Buffer"),
            size: (width * height * std::mem::size_of::<[f32;4]>()) as u64,
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

    // Helpers for min-neighbour + erosion GPU passes
    // Resize the min_elev_buffer (call after initialize_buffer).
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
                bg_entry!(3,&self.erosion_log_buffer),
            ],
        });
    }

    pub fn download_ocean_outflows(&self, height: usize) -> Vec<OutGpu> {
        let size_bytes = (height * std::mem::size_of::<OutGpu>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Outflow Staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Ocean Outflow DL") });
        encoder.copy_buffer_to_buffer(&self.ocean_out_buffer, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vec: Vec<OutGpu> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        vec
    }

    pub fn download_erosion_log(&self, width: usize, height: usize) -> Vec<[f32;4]> {
        let size_bytes = (width * height * std::mem::size_of::<[f32;4]>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Erosion Log Staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Erosion Log DL") });
        encoder.copy_buffer_to_buffer(&self.erosion_log_buffer, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vec: Vec<[f32;4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        vec
    }

    // TODO: Only used in let_slopes_settle, seems ripe for refactoring.
    pub fn run_repose_step(&mut self, width: usize, height: usize) {
        let consts = [width as f32, height as f32, HEX_SIZE];
        self.queue.write_buffer(&self.repose_consts_buf, 0, bytemuck::cast_slice(&consts));

        let workgroup_size: u32 = 256;
        let total = (width * height) as u32;
        let groups = (total + workgroup_size - 1) / workgroup_size;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("ReposeEncoder")});
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("ReposeDeltasPass")});
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(groups,1,1);
        }
        {
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

    // Run all simulation steps in a single batched command encoder
    pub fn run_simulation_step_batched(&mut self, width: usize, height: usize, sea_level: f32, flow_factor: f32, max_flow: f32) {
        // Update all constant buffers first
        let total_cells = width * height;
        
        // Rainfall constants
        let rain_constants = [total_cells as f32, sea_level, EVAPORATION_FACTOR, WIDTH_HEXAGONS as f32, (TOTAL_SEA_WIDTH + NORTH_DESERT_WIDTH) as f32, CONTINENTAL_SHELF_DEPTH];
        self.queue.write_buffer(&self.rain_constants_buffer, 0, bytemuck::cast_slice(&rain_constants));
        
        // Water routing constants
        let routing_constants = [width as f32, height as f32, flow_factor, max_flow];
        self.queue.write_buffer(&self.routing_constants_buffer, 0, bytemuck::cast_slice(&routing_constants));
        
        // Scatter constants - only write 2 f32s as the buffer was created for [f32; 2]
        let scatter_constants = [width as f32, height as f32];
        self.queue.write_buffer(&self.scatter_consts_buffer, 0, bytemuck::cast_slice(&scatter_constants));
        
        // Min neighbor constants
        let min_constants = [width as f32, height as f32];
        self.queue.write_buffer(&self.min_consts_buf, 0, bytemuck::cast_slice(&min_constants));
        
        // Erosion parameters
        let erosion_params: [f32; 6] = [KC, KE, KD, MAX_SLOPE, MAX_ELEVATION, HEX_SIZE];
        self.queue.write_buffer(&self.erosion_params, 0, bytemuck::cast_slice(&erosion_params));
        
        // Ocean boundary parameters
        let ocean_params = [sea_level, height as f32, width as f32, ABYSSAL_PLAINS_MAX_DEPTH];
        self.queue.write_buffer(&self.ocean_params_buffer, 0, bytemuck::cast_slice(&ocean_params));
        
        // Repose constants
        let repose_consts = [width as f32, height as f32, HEX_SIZE];
        self.queue.write_buffer(&self.repose_consts_buf, 0, bytemuck::cast_slice(&repose_consts));

        // Calculate dispatch sizes
        let workgroup_size: u32 = 256;
        let total = (width * height) as u32;
        let dispatch_x = (total + workgroup_size - 1) / workgroup_size;
        let ocean_dispatch = (height as u32 + workgroup_size - 1) / workgroup_size;

        // Create single command encoder for all passes
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batched Simulation Encoder"),
        });

        // 1. Rainfall pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rainfall Pass"),
            });
            pass.set_pipeline(&self.rainfall_pipeline);
            pass.set_bind_group(0, &self.rainfall_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 2. Min neighbor pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Min Neighbor Pass"),
            });
            pass.set_pipeline(&self.min_neigh_pipeline);
            pass.set_bind_group(0, &self.min_neigh_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 3. Erosion pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Erosion Pass"),
            });
            pass.set_pipeline(&self.erosion_pipeline);
            pass.set_bind_group(0, &self.erosion_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 4. Initialize water step (copy water/load from hex_data to next buffers AFTER erosion)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Init Water Pass"),
            });
            pass.set_pipeline(&self.init_water_pipeline);
            pass.set_bind_group(0, &self.init_water_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 5. Water routing pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water Routing Pass"),
            });
            pass.set_pipeline(&self.routing_pipeline);
            pass.set_bind_group(0, &self.routing_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 6. Scatter pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scatter Pass"),
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 7. Repose deltas pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Repose Deltas Pass"),
            });
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 8. Apply deltas pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Deltas Pass"),
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 9. Ocean boundary pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ocean Boundary Pass"),
            });
            pass.set_pipeline(&self.ocean_pipeline);
            pass.set_bind_group(0, &self.ocean_bind_group, &[]);
            pass.dispatch_workgroups(ocean_dispatch, 1, 1);
        }

        // Submit all passes at once
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}