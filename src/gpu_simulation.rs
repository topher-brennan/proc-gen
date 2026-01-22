use crate::constants::*;
use bytemuck::{Pod, Zeroable};

use wgpu::BufferUsages as BU;

macro_rules! buf_rw {
    ($binding:expr, $read_only:expr) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: $read_only,
                },
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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HexGpu {
    pub elevation: f32,
    pub residual_elevation: f32,
    pub water_depth: f32,
    pub residual_water_depth: f32,
    pub suspended_load: f32,
    pub residual_suspended_load: f32,
    pub rainfall: f32,
    pub erosion_multiplier: f32,
    pub uplift: f32,
}

pub struct GpuSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    hex_buffer: wgpu::Buffer,
    hex_buffer_size: usize,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // Rainfall
    rainfall_pipeline: wgpu::ComputePipeline,
    rainfall_bind_group_layout: wgpu::BindGroupLayout,
    rainfall_bind_group: wgpu::BindGroup,
    rain_params_buffer: wgpu::Buffer,
    // Water routing (gather pattern)
    routing_pipeline: wgpu::ComputePipeline,
    routing_bind_group_layout: wgpu::BindGroupLayout,
    routing_bind_group: wgpu::BindGroup,
    next_hex_buffer: wgpu::Buffer,
    // Min neighbor + flow target
    min_elev_buffer: wgpu::Buffer,
    flow_target_buffer: wgpu::Buffer,
    min_neigh_pipeline: wgpu::ComputePipeline,
    min_neigh_bind: wgpu::BindGroup,
    min_layout: wgpu::BindGroupLayout,
    // Erosion
    erosion_pipeline: wgpu::ComputePipeline,
    erosion_bind: wgpu::BindGroup,
    eros_layout: wgpu::BindGroupLayout,
    // Ocean boundary
    ocean_pipeline: wgpu::ComputePipeline,
    ocean_bind_group_layout: wgpu::BindGroupLayout,
    ocean_bind_group: wgpu::BindGroup,
    ocean_params_buffer: wgpu::Buffer,
    // Repose
    delta_buffer: wgpu::Buffer,
    repose_pipeline: wgpu::ComputePipeline,
    repose_bgl: wgpu::BindGroupLayout,
    repose_bind: wgpu::BindGroup,
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
            entries: &[buf_rw!(0, false)],
        });

        // Rainfall pipeline
        let rainfall_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rainfall Shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/rainfall.wgsl")
                )
                .into(),
            ),
        });

        let rainfall_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rainfall BGL"),
                entries: &[buf_rw!(0, false), buf_rw!(1, true), uniform_entry!(2)],
            });

        let rainfall_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        // Water routing pipeline (gather pattern with flow_target)
        let routing_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Routing Shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/water_routing.wgsl")
                )
                .into(),
            ),
        });

        let routing_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Routing BGL"),
                entries: &[
                    buf_rw!(0, true),  // hex_data (read)
                    buf_rw!(1, false), // next_hex_data (write)
                    buf_rw!(2, true),  // flow_target (read)
                    buf_rw!(3, true),  // min_elev (read)
                ],
            });

        let routing_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        // Create buffers
        let hex_buffer_size = std::mem::size_of::<HexGpu>();
        let hex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: hex_buffer_size as u64,
            usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_hex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Hex Buffer"),
            size: hex_buffer_size as u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        let rain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rainfall Params Buffer"),
            size: (2 * std::mem::size_of::<f32>()) as u64, // sea_level + seasonal_rain_multiplier
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let min_elev_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size: 4,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        let flow_target_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow_target"),
            size: 4,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind groups (will be recreated with proper sizes in initialize_buffer)
        let routing_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &routing_bind_group_layout,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &next_hex_buffer),
                bg_entry!(2, &flow_target_buffer),
                bg_entry!(3, &min_elev_buffer),
            ],
        });

        let rainfall_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rainfall Bind Group"),
            layout: &rainfall_bind_group_layout,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &min_elev_buffer),
                bg_entry!(2, &rain_params_buffer),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[bg_entry!(0, &hex_buffer)],
        });

        // Min neighbor pipeline (now outputs both min_elev and flow_target)
        let min_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("min shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/min_neigh.wgsl")
                )
                .into(),
            ),
        });

        let min_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("min BGL"),
            entries: &[
                buf_rw!(0, true),  // hex_data
                buf_rw!(1, false), // min_elev
                buf_rw!(2, false), // flow_target
            ],
        });

        let min_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("min layout"),
            bind_group_layouts: &[&min_bgl],
            push_constant_ranges: &[],
        });

        let min_neigh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("min pipe"),
            layout: Some(&min_layout),
            module: &min_shader,
            entry_point: "main",
        });

        let min_neigh_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("min BG"),
            layout: &min_bgl,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &min_elev_buffer),
                bg_entry!(2, &flow_target_buffer),
            ],
        });

        // Erosion pipeline
        let eros_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("erosion shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/erosion.wgsl")
                )
                .into(),
            ),
        });

        let eros_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion BGL"),
            entries: &[
                buf_rw!(0, false),
                buf_rw!(1, true),
                uniform_entry!(2),
            ],
        });

        let eros_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eros layout"),
            bind_group_layouts: &[&eros_bgl],
            push_constant_ranges: &[],
        });

        let erosion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eros pipe"),
            layout: Some(&eros_layout),
            module: &eros_shader,
            entry_point: "main",
        });

        let ocean_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean Params Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let erosion_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eros BG"),
            layout: &eros_bgl,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &min_elev_buffer),
                bg_entry!(2, &ocean_params_buffer),
            ],
        });

        // Ocean boundary pipeline
        let ocean_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ocean Boundary Shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/ocean_boundary.wgsl")
                )
                .into(),
            ),
        });

        let ocean_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Ocean Boundary BGL"),
                entries: &[buf_rw!(0, false), uniform_entry!(1)],
            });

        let ocean_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        let ocean_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &ocean_bind_group_layout,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &ocean_params_buffer),
            ],
        });

        // Repose pipeline
        let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: 4,
            usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let repose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Repose Shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/repose_deltas.wgsl")
                )
                .into(),
            ),
        });

        let repose_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Repose BGL"),
            entries: &[buf_rw!(0, true), buf_rw!(1, false)],
        });

        let repose_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        let repose_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Repose Bind Group"),
            layout: &repose_bgl,
            entries: &[bg_entry!(0, &hex_buffer), bg_entry!(1, &delta_buffer)],
        });

        let apply_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/generated_constants.wgsl"),
                    include_str!("shaders/common.wgsl"),
                    include_str!("shaders/apply_deltas.wgsl")
                )
                .into(),
            ),
        });

        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Apply BGL"),
            entries: &[buf_rw!(0, false), buf_rw!(1, false)],
        });

        let apply_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
            entries: &[bg_entry!(0, &hex_buffer), bg_entry!(1, &delta_buffer)],
        });

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
            rain_params_buffer,
            routing_pipeline,
            routing_bind_group_layout,
            routing_bind_group,
            next_hex_buffer,
            min_elev_buffer,
            flow_target_buffer,
            min_neigh_pipeline,
            min_neigh_bind,
            min_layout: min_bgl,
            erosion_pipeline,
            erosion_bind,
            eros_layout: eros_bgl,
            ocean_pipeline,
            ocean_bind_group_layout,
            ocean_bind_group,
            ocean_params_buffer,
            delta_buffer,
            repose_pipeline,
            repose_bgl,
            repose_bind,
            apply_pipeline,
            apply_bgl,
            apply_bind,
        }
    }

    pub fn sync_device(&self) {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sync Encoder"),
            });
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn initialize_buffer(&mut self, width: usize, height: usize) {
        let buffer_size = width * height * std::mem::size_of::<HexGpu>();
        self.hex_buffer_size = buffer_size;

        self.hex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: buffer_size as u64,
            usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        self.next_hex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Hex Buffer"),
            size: buffer_size as u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[bg_entry!(0, &self.hex_buffer)],
        });

        self.ocean_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &self.ocean_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.ocean_params_buffer),
            ],
        });

        self.resize_min_buffers(width, height);

        let delta_bytes = (width * height * std::mem::size_of::<u32>()) as u64;
        self.delta_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: delta_bytes,
            usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
            mapped_at_creation: false,
        });

        self.repose_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Repose Bind Group"),
            layout: &self.repose_bgl,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.delta_buffer),
            ],
        });

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
        self.queue
            .write_buffer(&self.hex_buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn download_data(&self) -> Vec<HexGpu> {
        let buffer_size = self.hex_buffer_size;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: BU::MAP_READ | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

    pub fn resize_min_buffers(&mut self, width: usize, height: usize) {
        let size_f32 = (width * height * std::mem::size_of::<f32>()) as u64;
        let size_u32 = (width * height * std::mem::size_of::<u32>()) as u64;

        self.min_elev_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size: size_f32,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        self.flow_target_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow_target"),
            size: size_u32,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        self.min_neigh_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("min BG"),
            layout: &self.min_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.min_elev_buffer),
                bg_entry!(2, &self.flow_target_buffer),
            ],
        });

        self.erosion_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eros BG"),
            layout: &self.eros_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.min_elev_buffer),
                bg_entry!(2, &self.ocean_params_buffer),
            ],
        });

        self.rainfall_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rainfall Bind Group"),
            layout: &self.rainfall_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.min_elev_buffer),
                bg_entry!(2, &self.rain_params_buffer),
            ],
        });

        self.routing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &self.routing_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.next_hex_buffer),
                bg_entry!(2, &self.flow_target_buffer),
                bg_entry!(3, &self.min_elev_buffer),
            ],
        });
    }

    pub fn run_repose_pass(&self, _width: usize, _height: usize) {
        let total = (WIDTH_HEXAGONS * HEIGHT_PIXELS) as u32;
        let groups = (total + 255) / 256;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ReposeEncoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReposeDeltasPass"),
            });
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ApplyDeltasPass"),
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn download_hex_data(&self) -> Vec<HexGpu> {
        self.download_data()
    }

    pub fn run_simulation_step_batched(
        &mut self,
        _width: usize,
        _height: usize,
        sea_level: f32,
        seasonal_rain_multiplier: f32,
    ) {
        self.queue.write_buffer(
            &self.rain_params_buffer,
            0,
            bytemuck::cast_slice(&[sea_level, seasonal_rain_multiplier]),
        );
        self.queue.write_buffer(
            &self.ocean_params_buffer,
            0,
            bytemuck::cast_slice(&[sea_level]),
        );

        let workgroup_size: u32 = 256;
        let total = (WIDTH_HEXAGONS * HEIGHT_PIXELS) as u32;
        let dispatch_x = (total + workgroup_size - 1) / workgroup_size;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched Simulation Encoder"),
            });

        // 1. Rainfall
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rainfall Pass"),
            });
            pass.set_pipeline(&self.rainfall_pipeline);
            pass.set_bind_group(0, &self.rainfall_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 2. Min neighbor + flow target
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Min Neighbor Pass"),
            });
            pass.set_pipeline(&self.min_neigh_pipeline);
            pass.set_bind_group(0, &self.min_neigh_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 3. Erosion
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Erosion Pass"),
            });
            pass.set_pipeline(&self.erosion_pipeline);
            pass.set_bind_group(0, &self.erosion_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 4. Water routing (gather pattern using precomputed flow_target)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water Routing Pass"),
            });
            pass.set_pipeline(&self.routing_pipeline);
            pass.set_bind_group(0, &self.routing_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 5. Copy (use GPU-native buffer copy - faster than compute shader)
        encoder.copy_buffer_to_buffer(
            &self.next_hex_buffer,
            0,
            &self.hex_buffer,
            0,
            self.hex_buffer_size as u64,
        );

        // 6. Repose deltas
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Repose Deltas Pass"),
            });
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 7. Apply deltas
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Deltas Pass"),
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 8. Ocean boundary
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ocean Boundary Pass"),
            });
            pass.set_pipeline(&self.ocean_pipeline);
            pass.set_bind_group(0, &self.ocean_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
