use crate::constants::*;
use bytemuck::{Pod, Zeroable};

// Helper macros for concise bind-group definitions
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

macro_rules! dispatch_compute {
    ($device:expr, $queue:expr, $pipeline:expr, $bind_group:expr, $invocations:expr) => {{
        let mut encoder = $device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute pass"),
            });
            pass.set_pipeline(&$pipeline);
            pass.set_bind_group(0, &$bind_group, &[]);
            pass.dispatch_workgroups($invocations, 1, 1);
        }
        $queue.submit(std::iter::once(encoder.finish()));
    }};
}

/*
    elevation: f32,
    elevation_residual: f32,
    water_depth: f32,
    water_depth_residual: f32,
    suspended_load: f32,
    suspended_load_residual: f32,
    rainfall: f32,
    erosion_multiplier: f32,
    uplift: f32,
*/

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HexGpu {
    pub elevation: f32,
    pub residual_elevation: f32,
    pub water_depth: f32,
    pub residual_water_depth: f32,
    pub suspended_load: f32,
    pub residual_suspended_load: f32,
    pub rainfall: f32, // per-hex rainfall depth per step
    pub erosion_multiplier: f32,
    pub uplift: f32,
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
    rain_params_buffer: wgpu::Buffer, // Only seasonal_rain_multiplier now
    // Water-routing resources (gather pattern - no atomics!)
    routing_pipeline: wgpu::ComputePipeline,
    routing_bind_group_layout: wgpu::BindGroupLayout,
    routing_bind_group: wgpu::BindGroup,
    next_hex_buffer: wgpu::Buffer, // Double-buffer for gather pattern
    flow_target_buffer: wgpu::Buffer, // u32 index of flow target
    // Copy pass (copies next_hex_buffer back to hex_buffer)
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group: wgpu::BindGroup,
    // Min neighbor pass (computes flow_target and min_elev)
    min_elev_buffer: wgpu::Buffer,
    min_neigh_pipeline: wgpu::ComputePipeline,
    min_neigh_bind: wgpu::BindGroup,
    min_layout: wgpu::BindGroupLayout,
    // Erosion resources
    erosion_pipeline: wgpu::ComputePipeline,
    erosion_bind: wgpu::BindGroup,
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
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // TODO: Remove?
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
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
                    // min_elev buffer (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Uniform buffer with constants
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        // Water routing pipeline (gather pattern - no atomics!)
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
                    buf_rw!(0, true),  // hex_data (read-only)
                    buf_rw!(1, true),  // flow_target (read-only)
                    buf_rw!(2, false), // next_hex_data (write)
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

        // Copy pipeline (copies next_hex_data back to hex_data)
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/copy_hex_data.wgsl").into()),
        });

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Copy BGL"),
                entries: &[
                    buf_rw!(0, false), // hex_data (write)
                    buf_rw!(1, true),  // next_hex_data (read)
                ],
            });

        let copy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Copy Pipeline Layout"),
                bind_group_layouts: &[&copy_bind_group_layout],
                push_constant_ranges: &[],
            });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Copy Pipeline"),
            layout: Some(&copy_pipeline_layout),
            module: &copy_shader,
            entry_point: "main",
        });

        // Create buffers with minimal non-zero size (overwritten later)
        let hex_buffer_size = std::mem::size_of::<HexGpu>();
        let hex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: hex_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Runtime params buffer - only seasonal_rain_multiplier now (other constants are overrides)
        let rain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rainfall Params Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- min_elev_buffer and flow_target_buffer (created early for bind groups) ----
        let min_elev_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size: 4, // resized later
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let flow_target_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow_target"),
            size: 4, // resized later
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_hex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Hex Buffer"),
            size: std::mem::size_of::<HexGpu>() as u64, // resized later
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let routing_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &routing_bind_group_layout,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &flow_target_buffer),
                bg_entry!(2, &next_hex_buffer),
            ],
        });

        let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Copy Bind Group"),
            layout: &copy_bind_group_layout,
            entries: &[
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &next_hex_buffer),
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
                    resource: min_elev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rain_params_buffer.as_entire_binding(),
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

        // ---- min_neigh pipeline ----
        // Computes both flow_target (for water routing) and min_elev (for erosion)
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
                buf_rw!(0, true),  // hex_data (read)
                buf_rw!(1, false), // flow_target (write)
                buf_rw!(2, false), // min_elev (write)
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
                bg_entry!(1, &flow_target_buffer),
                bg_entry!(2, &min_elev_buffer),
            ],
        });

        // ---- erosion pipeline ----
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
                buf_rw!(0, false), // hex_data
                buf_rw!(1, true),  // min_elev (read)
                buf_rw!(2, false), // erosion log buffer
                uniform_entry!(3), // RuntimeParams (sea_level) - shared with ocean_boundary
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
        // Create a placeholder erosion_log_buffer before binding (resized later)
        let erosion_log_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Erosion Log Buffer"),
            size: 4u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });

        // RuntimeParams buffer (sea_level) - shared between erosion and ocean_boundary shaders
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
                bg_entry!(2, &erosion_log_buffer),
                bg_entry!(3, &ocean_params_buffer),
            ],
        });

        // ---- ocean boundary pipeline ----
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
                entries: &[
                    buf_rw!(0, false), // hex_data
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // params (sea_level)
                    buf_rw!(2, false), // outflow buffer
                ],
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

        // ocean_params_buffer is created earlier (shared with erosion shader)

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
                bg_entry!(0, &hex_buffer),
                bg_entry!(1, &ocean_params_buffer),
                bg_entry!(2, &ocean_out_buffer),
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
            entries: &[
                buf_rw!(0, true),  // hex_data (read-only)
                buf_rw!(1, false), // delta buffer (atomic read-write)
            ],
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/apply_deltas.wgsl").into()),
        });

        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Apply BGL"),
            entries: &[
                buf_rw!(0, false), // hex_data read-write
                buf_rw!(1, false), // delta buffer read-write
            ],
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
            rain_params_buffer,
            // Water routing (gather pattern)
            routing_pipeline,
            routing_bind_group_layout,
            routing_bind_group,
            next_hex_buffer,
            flow_target_buffer,
            // Copy pass
            copy_pipeline,
            copy_bind_group_layout,
            copy_bind_group,
            // Min neighbor
            min_elev_buffer,
            min_neigh_pipeline,
            min_neigh_bind,
            min_layout: min_layout_clone,
            // Erosion
            erosion_pipeline,
            erosion_bind,
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
            apply_pipeline,
            apply_bgl,
            apply_bind,
        }
    }

    /// Lightweight keep-alive to service the device without blocking.
    /// Uses non-blocking poll - good for calling frequently.
    pub fn heartbeat(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// Synchronize with the GPU by waiting for all submitted work to complete.
    /// This is much cheaper than downloading data - it just waits without copying.
    /// Use this to prevent "Parent device is lost" errors without the overhead
    /// of downloading large buffers.
    pub fn sync_device(&self) {
        // Submit an empty command buffer to ensure there's work to wait on
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sync Encoder"),
            });
        self.queue.submit(std::iter::once(encoder.finish()));
        // Block until all GPU work completes
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn initialize_buffer(&mut self, width: usize, height: usize) {
        let buffer_size = width * height * std::mem::size_of::<HexGpu>();
        self.hex_buffer_size = buffer_size;

        self.hex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hex Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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

        // Note: rainfall_bind_group is recreated in resize_min_buffers()

        // Resize next_hex_buffer for gather pattern double-buffering
        self.next_hex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Next Hex Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Note: routing_bind_group and copy_bind_group are recreated in resize_min_buffers()
        // since they depend on flow_target_buffer

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
            size: (width * height * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.ocean_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ocean Boundary BG"),
            layout: &self.ocean_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.ocean_params_buffer),
                bg_entry!(2, &self.ocean_out_buffer),
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
        self.queue
            .write_buffer(&self.hex_buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn download_data(&self) -> Vec<HexGpu> {
        let buffer_size = self.hex_buffer_size;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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

    // Helpers for min-neighbour + erosion + water routing GPU passes
    // Resize the min_elev_buffer and flow_target_buffer (call after initialize_buffer).
    pub fn resize_min_buffers(&mut self, width: usize, height: usize) {
        let f32_size = (width * height * std::mem::size_of::<f32>()) as u64;
        let u32_size = (width * height * std::mem::size_of::<u32>()) as u64;
        
        self.min_elev_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("min_elev"),
            size: f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.flow_target_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow_target"),
            size: u32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.min_neigh_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("min BG"),
            layout: &self.min_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.flow_target_buffer),
                bg_entry!(2, &self.min_elev_buffer),
            ],
        });

        self.erosion_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eros BG"),
            layout: &self.eros_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.min_elev_buffer),
                bg_entry!(2, &self.erosion_log_buffer),
                bg_entry!(3, &self.ocean_params_buffer),
            ],
        });

        // Recreate rainfall bind group with resized min_elev_buffer
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
                    resource: self.min_elev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rain_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate routing bind group (gather pattern)
        self.routing_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Routing Bind Group"),
            layout: &self.routing_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.flow_target_buffer),
                bg_entry!(2, &self.next_hex_buffer),
            ],
        });

        // Recreate copy bind group
        self.copy_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Copy Bind Group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                bg_entry!(0, &self.hex_buffer),
                bg_entry!(1, &self.next_hex_buffer),
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ocean Outflow DL"),
            });
        encoder.copy_buffer_to_buffer(&self.ocean_out_buffer, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vec: Vec<OutGpu> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        vec
    }

    pub fn download_erosion_log(&self, width: usize, height: usize) -> Vec<[f32; 4]> {
        let size_bytes = (width * height * std::mem::size_of::<[f32; 4]>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Erosion Log Staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Erosion Log DL"),
            });
        encoder.copy_buffer_to_buffer(&self.erosion_log_buffer, 0, &staging, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vec: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        vec
    }

    // TODO: Only used in let_slopes_settle, seems ripe for refactoring.
    pub fn run_repose_step(&mut self, width: usize, height: usize) {
        let workgroup_size: u32 = 256;
        let total = (width * height) as u32;
        let groups = (total + workgroup_size - 1) / workgroup_size;

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

    // Run all simulation steps in a single batched command encoder
    // Note: width, height, sea_level are now override constants baked into shaders
    pub fn run_simulation_step_batched(
        &mut self,
        _width: usize,
        _height: usize,
        sea_level: f32,
        seasonal_rain_multiplier: f32,
    ) {
        // Runtime parameters
        self.queue.write_buffer(
            &self.rain_params_buffer,
            0,
            bytemuck::cast_slice(&[seasonal_rain_multiplier]),
        );
        self.queue.write_buffer(
            &self.ocean_params_buffer,
            0,
            bytemuck::cast_slice(&[sea_level]),
        );

        // Calculate dispatch sizes using compile-time constants
        let workgroup_size: u32 = 256;
        let total = (WIDTH_HEXAGONS * HEIGHT_PIXELS) as u32;
        let dispatch_x = (total + workgroup_size - 1) / workgroup_size;
        // Ocean boundary needs to process all cells to find west/south edge cells
        let ocean_dispatch = dispatch_x;

        // Create single command encoder for all passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        // 4. Water routing pass (gather pattern - reads hex_data, writes to next_hex_data)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water Routing Pass"),
            });
            pass.set_pipeline(&self.routing_pipeline);
            pass.set_bind_group(0, &self.routing_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 5. Copy pass (copies next_hex_data back to hex_data)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy Pass"),
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, &self.copy_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 6. Repose deltas pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Repose Deltas Pass"),
            });
            pass.set_pipeline(&self.repose_pipeline);
            pass.set_bind_group(0, &self.repose_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 7. Apply deltas pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Deltas Pass"),
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &self.apply_bind, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        // 8. Ocean boundary pass
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
