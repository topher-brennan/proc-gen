mod gpu_simulation;
use gpu_simulation::{GpuSimulation, HexGpu};
use pollster;

fn main() {
    // Initialize GPU simulation
    let mut gpu_sim = pollster::block_on(GpuSimulation::new());
    
    // Test with a small 10x10 grid
    let width = 10;
    let height = 10;
    let total_cells = width * height;
    
    // Initialize buffer
    gpu_sim.initialize_buffer(width, height);
    
    // Create test data
    let mut test_data = Vec::with_capacity(total_cells);
    for i in 0..total_cells {
        test_data.push(HexGpu {
            elevation: i as f32 * 10.0,
            water_depth: 0.0,
            suspended_load: 0.0,
            _padding: 0.0,
        });
    }
    
    // Upload data to GPU
    gpu_sim.upload_data(&test_data);
    
    // Run one rainfall step using the new GPU compute pass
    gpu_sim.run_rainfall_step(0.00001, total_cells); // use same scale as CPU RAIN_PER_STEP
    
    // Download results
    let results = gpu_sim.download_data();
    
    // Print some results to verify it worked
    println!("GPU simulation test completed!");
    println!("Original water depth at cell 0: {}", test_data[0].water_depth);
    println!("New water depth at cell 0: {}", results[0].water_depth);
    println!("Original water depth at cell 5: {}", test_data[5].water_depth);
    println!("New water depth at cell 5: {}", results[5].water_depth);
} 