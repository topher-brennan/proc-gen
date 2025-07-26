use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HexGpu {
    pub elevation: f32,
    pub water_depth: f32,
    pub suspended_load: f32,
    pub _padding: f32, // Ensure 16-byte alignment
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
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
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/water_routing.wgsl"))),
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
            size: std::mem::size_of::<[f32; 2]>() as u64,
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
    pub fn run_rainfall_step(&mut self, rain_per_step: f32, total_cells: usize) {
        // Update constants buffer
        let constants = [rain_per_step, total_cells as f32];
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
    pub fn download_routing_results(&self) -> (Vec<f32>, Vec<f32>) {
        let buf_size = self.hex_buffer_size / std::mem::size_of::<HexGpu>() * std::mem::size_of::<f32>();

        let create_staging = |label: &str| self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(label),
            size: buf_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });

        let staging_water = create_staging("StageWater");
        let staging_load  = create_staging("StageLoad");

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("RouteDownloadEnc")});
        encoder.copy_buffer_to_buffer(&self.next_water_buffer,0,&staging_water,0,buf_size as u64);
        encoder.copy_buffer_to_buffer(&self.next_load_buffer ,0,&staging_load ,0,buf_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let read_buffer = |buf: &wgpu::Buffer| {
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

        (read_buffer(&staging_water), read_buffer(&staging_load))
    }
} 