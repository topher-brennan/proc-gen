We want to simulate erosion and deposition on a region of a ficticious continent on an Earth-like planet.

The region will be assumed to be 1080 in north-south extent and roughly 1920 miles in east-west extent, so that on a typical large monitor each pixel will represent a half-mile. Furthermore the region will be assumed to be centered on a point 540 miles south of the 25th parallel north. I *think* that should limit distortions from treating the region as a flat rectangle.

Starting point for the erosion simulation will mostly be non-naturalistic, with maybe some noise.

Try to force a river to pass through a point approximately 300 miles (600 hexes) from the northern edge of the map. We might need raised elevation to the north / south to force river going through there. Maybe pick points +/-70 miles from the forced-river point (230/370 miles from north edge of map), assign them an elevation of 2100 feet, declining 30 feet/mile in north/south directions. Might decline 3 feet/mile to the east, 300 feet/mile to the west? The idea would be very rapid decline towards the sea, much slower decline until it merges with the higher eastern elevations. Or just not decline at all into the east.

Memory considerations for design: I'm developing this on a PC with 16GB ram. Closing programs and looking at the Task Manager, I don't seem to be able to get below about 6GM memory in use. Maybe I could get more aggressive about killing unnecessary background processes, but until them I've got a hard limit on amount of ram we can use of 10GB. Currently planning for almost 10 million hexes, so I'd need to limit myself to 1KB per hex.

Other system notes: I've got a 12GB graphics card with a NVIDIA GPU that could maybe be leveraged?

## Experimental Notes

When first implementing rain, I found it took a surprisingly long time for water to reach equilibrium. With a map 148 hexes wide, it takes ~82,000 steps for mean water level to reach the point where ~99% of water coming in is going out, and not fully stabilized after 148,000 steps. With no rain and the only water coming from a single "river source" hex, my impression was that it stabilizes sooner, after around maybe 23,000 steps, but that *might* be an illusion created by rounding errors and the fact that having all water come in at a single point creates more room for randomness to create temporary plateaus.

My hope is that when errosion is implemented stabilization will be faster.

Update: stabilization doesn't seem faster with errosion. With a map 296 hexes wide, takes almost ~100,000 steps to stabilize. Want a model that allows for 8,000+ feet of errosion in some reasonable timeframe.

Update 2: With HEIGHT_PIXELS = 216 and WIDTH_PIXELS = 384, 10 rounds took ~380s.

## High-Level Strategy: Sea-to-River Erosion & Deposition Model

The goal is to evolve a hex-grid terrain until realistic landforms emerge under the combined influence of:
1. A fixed-level sea along the western border (open boundary with zero-gradient water surface).
2. A major sediment-laden river entering from the eastern edge.
3. Spatially/temporally variable rainfall that spawns distributed minor streams.
4. Gravity-driven surface flow that erodes, transports, and deposits sediment.

Key building blocks
-------------------
1. **State stored per hex**
   • elevation (bedrock + alluvium depth)
   • water depth & discharge vector
   • suspended sediment mass & grain-size fractions (≥1 coarse + ≤1 fine/cohesive bucket)
   • surface type flag (dry / river / lake / sea)

2. **Hydrology step** (per Δt)
   a. Add boundary inputs – fixed sea level on west, river influx on east, and rainfall over the field.
   b. Route water using a distributed flow algorithm (D8-style but adapted to our axial hex neighbours).  Options:
      – Instantaneous steepest-descent routing for smaller Δt.
      – Full shallow-water solver (e.g. 2-D Saint-Venant) if we can afford it.
   c. Allow overtopping; if no downslope exit exists, raise a transient lake level until outflow path opens.

3. **Sediment transport & morphodynamics**
   • Compute sediment capacity C = k · Qᵐ · Sⁿ  (where Q = discharge, S = local slope, m≈1, n≈1–1.5).
   • If suspended load < C: erode bed (rate ∝ (C-load) · f(resistance)).  Resistance rises sharply for cohesive clay.
   • If load > C: deposit the surplus.  Store coarse first, clay last (helps natural levee & delta stratigraphy).
   • Update elevation; mass-balance must hold.

4. **Sea interaction / Delta rules**
   • Western sea cells keep elevation fixed at 0 ft (or user-defined mean sea level).  Incoming sediment that strikes a sea cell is deposited, building subaerial delta as bed rises above 0.
   • Apply a velocity-drop factor when water enters a sea cell to mimic energy loss; this encourages mouth-bar formation, bifurcation, and eventual delta lobe switching similar to [DeltaRCM].
   • Wave/tide smoothing can be approximated by lateral diffusion of coastal bathymetry or an explicit along-shore sediment flux.

5. **Rainfall driver**
   • Time-varying stochastic field (Perlin / simplex noise) producing realistic storm tracks.
   • Convert mm h⁻¹ to discharge increment per hex; infiltration can be a simple exponential decay with soil depth.

6. **Numerical scheme & performance**
   • Use fixed-size arrays; single-precision floats are adequate → ~10 million hexes ≈ <1 GB/core.
   • Sub-cycle water routing multiple times per morphologic step to maintain stability.
   • Parallelise over rows (Rayon) or explore GPU kernels later.

7. **Validation hooks**
   • Track global sediment mass, sea-level flux balance.
   • Metrics: drainage density, slope-area relation, hypsometry, delta shore rugosity – compare to natural analogues (Mississippi, Nile, Wax Lake, etc.).

Suggested incremental milestones
--------------------------------
Our working roadmap is intentionally incremental so each stage can be tested in a few-minute run.

TODO: At some point we need to implement having an ocean at the west end of the map, rather than just conveying where the ocean should be through colorization. 

1. **Hydrology sanity-check (done)**  
   Uniform rainfall over the whole domain + a fixed sea along the west edge.  
   Goal: make sure water routes to the coast and the mass balance (rain = out-flow + Δstorage) stays within floating-point error.

2. **Add controlled inflow (next up)**  
   Inject a steady discharge at one or two cells on the east edge to represent a major river.  
   Goal: confirm the hydrograph (rain + river = out-flow).

3. **Minimal erosion / deposition**  
   Add a single suspended-load scalar per hex and use a simple capacity rule
   `capacity = k_c * discharge * slope`, with erosion if load<capacity and deposition otherwise.  
   Goal: see channels incise into the synthetic slope and tiny fans build at the coast.

4. **Two-fraction sediment & cohesion**  
   Split the load into “sand” and “mud”, give mud a higher erosion threshold so it travels farther.  
   Goal: natural levees, mouth-bar bifurcation, delta progradation that depends on grain size.

5. **Lake formation / ponding (optional)**  
   Implement a priority-flood depression fill so enclosed basins spill once the water surface reaches the lowest saddle.

Each stage keeps run-time and memory roughly constant; only new per-hex fields are added when necessary.

Reading list
------------
• [Liang et al. 2015](https://esurf.copernicus.org/articles/3/67/2015/esurf-3-67-2015.html) – DeltaRCM reduced-complexity delta model ([GitHub](https://github.com/DeltaRCM/pyDeltaRCM)).
• Caldwell & Edmonds 2014, "The effects of sediment properties on deltaic processes and morphologies: A numerical modeling
study" and Edmonds & Slingerland 2010, "Significant effect of sediment cohesion on delta morphology" – grain-size & cohesion effects.
• [Nardin & Fagherazzi 2012](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012GL051788) – wave influence on mouth-bar evolution.
• Julien 2010, Erosion and Sedimentation, 2nd ed – sediment transport formulas for rivers.


Experimental Notes
------------------
Round 99: water in 1.282  stored 1050952  mean depth 0.21 ft  max depth 1090.52 ft  wet 215765 (4.3%)  source elevation 7470.32 ft  westernmost land X 2021
  min elevation: -10005.46 ft  max elevation: 18474.50 ft  time: 2252.2555236s
Rainfall simulation complete – steps: 443400, total outflow to sea: 0.00 ft-hexes,
 westernmost land: 2020, westernmost land north: 2031, westernmost land south: 2076,
 water remaining on land: 1042841.00 ft-hexes, water remaining north: 326685.81 ft-hexes, water remaining central: 220345.70 ft-hexes, sediment in 0.0,  sediment out 0.0

Performance optimization
------------------------
Recommendations from GPT-o3. (Human's note: o3's analysis suggest performance gains are not large enough to make me want to prioritize.)

Ran tool
Read file: src/gpu_simulation.rs

Read file: src/shaders/repose_deltas.wgsl

Read file: src/shaders/apply_deltas.wgsl

Read file: src/shaders/common.wgsl

Read file: src/shaders/scatter_water.wgsl

Ran tool
Read file: src/gpu_simulation.rs

Below are the hot‐spots I would look at first.  None of them require changing the high-level model of your simulation; they are all “mechanical” improvements that usually translate into double-digit speed-ups when the map is large.

────────────────────────────────────────
1.  Collapse passes that walk the entire grid
────────────────────────────────────────
Today you do 7–10 full-grid passes per frame:

• rainfall → • water-routing (gather) → • scatter  
• min_neigh → • erosion  
• repose_deltas → • apply_deltas  
(+ river-source, ocean boundary, etc.)

Every pass costs:

( A ) CPU overhead  
  ‒ queue.write_buffer for a few scalars  
  ‒ create_command_encoder + begin/end_pass + submit  
( B ) A full read & write of the hex buffer (≈ memory-bandwidth bound)

If you combine passes that already touch the same data the GPU work itself hardly increases, yet you cut (A)+(B) for each pair you fuse.

Good first fusions
a)  rain + water-routing (one read/write saved)  
   • Inside `route_water` just add the per-cell rainfall before you do anything else.

b)  min_neigh + erosion (one read/write + a whole storage buffer saved)  
   • While you loop over neighbours to find the minimum height, keep the value in a local
     variable and immediately compute `slope`/`capacity`.  You can then drop the
     `min_elev` buffer and the extra dispatch entirely.

c)  repose_deltas + apply_deltas (encoder + full pass removed)  
   • Instead of writing to `delta_buffer`, atomically modify the destination cell’s
     elevation in place (see §3 below).  That leaves you with just one dispatch.

Even fusing only (b) and (c) removes two full passes and one large buffer, which tends to
shave ~20-25 % off frame time on desktop GPUs.

────────────────────────────────────────
2.  Replace CAS loops with a single atomic add
────────────────────────────────────────
`repose_deltas.wgsl` uses compare-exchange loops to modify `delta_buffer`:

```wgsl
old_val = atomicLoad(&delta_buffer[idx]);
new_val = bitcast<u32>(bitcast<f32>(old_val) - excess);
atomicCompareExchangeWeak(&delta_buffer[idx], old_val, new_val);
```

Two problems:

1. Atomic CAS is **much** slower than atomic add.
2. You spin in a loop until the CAS succeeds, so contention can explode.

Quick fix  
Store the deltas in **integer fixed point** (e.g. millimetres = value*1000) and use the
built-in `atomicAdd` on `atomic<i32>`:

```wgsl
let excess_mm : i32 = i32(excess * 1000.0);
atomicAdd(&delta_buffer[idx]      , -excess_mm);
atomicAdd(&delta_buffer[neighbor] ,  excess_mm);
```

That removes the loop and almost all of the extra latency.  
When you later add the delta to `hex_data` just multiply by 1 / 1000.

Even better – after switching to `atomicAdd`, write straight to `hex_data[idx].elevation`
(also stored as an `atomic<i32>` in millimetres) and you can delete `delta_buffer`
and the second shader pass altogether.

────────────────────────────────────────
3.  Scrub unnecessary uniform-buffer traffic
────────────────────────────────────────
Every step updates a tiny `[f32; N]` uniform by calling `queue.write_buffer`.  That causes a
CPU-side allocation and a PCIe upload even though the data are < 64 B.

Options:

A.  Use **push constants** – supported on all desktop GPUs and on mobile
    Vulkan/OpenGL ES 3.1+.

B.  Keep one big “sim constants” buffer and write it **once** per frame, not per pass.

On discrete GPUs the gain is mostly CPU (fewer driver calls); on integrated GPUs it also
saves bandwidth.

────────────────────────────────────────
4.  Tune the work-group size
────────────────────────────────────────
You hard-code `@workgroup_size(256)`.  Ampere/RDNA GPUs usually peak at 128 or 64
threads/SM for memory-bound kernels, and Intel iGPUs often prefer 32.  Try compiling each
kernel with 64/128/256 and time them – the difference can be 10-20 %.

(You can keep one shader and select the work-group size with a `#define` or a WGSL
override constant that is specialised at pipeline-creation time.)

────────────────────────────────────────
5.  Water-routing:  single-pass “atomic scatter”
────────────────────────────────────────
The current gather/scatter dance:

1.  Per cell, decide where water should flow → write to three temporary buffers.  
2.  Second pass subtract outflow, iterate neighbours to collect inflow.

A single pass can do both with atomics:

```wgsl
// for each cell
if(move_w > 0.0) {
    atomicAdd(&hex_data[idx].water_depth, -move_w);
    atomicAdd(&hex_data[target].water_depth,  move_w);
    … same for suspended_load …
}
```

Again you can use fixed-point `i32` so the atomicAdd is legal.

When the physics allow it this removes two full-grid reads/writes plus three temporary
buffers.

────────────────────────────────────────
6.  Double-buffering the command encoder
────────────────────────────────────────
If you decide not to fuse passes, at least reuse one `wgpu::CommandEncoder`:

```rust
let mut enc = self.device.create_command_encoder(...);
{ /* rainfall */ }
{ /* routing */ }
{ /* scatter */ }
...
self.queue.submit(Some(enc.finish()));
```

Creating & submitting once per frame instead of once per pass knocks ~1 ms off the CPU
side on typical desktops.

────────────────────────────────────────
Where to start?
────────────────────────────────────────
Order by return-on-investment:

1. Swap the CAS loops in `repose_deltas` for `atomicAdd` (fast & easy).  
2. Fuse `min_neigh` + `erosion` (one kernel rewrite, one Rust call deleted).  
3. Fuse `repose_deltas` + `apply_deltas` (after step 1 this is trivial).  
4. Replace the tiny uniforms with push constants.  
5. Experiment with work-group sizes (compile-time constant, no API changes).  
6. Water-routing single-pass rewrite (largest gain, most effort).

Do (1)-(3) and you should see the “repose step” drop from ~17 % of frame time to < 8 %
and the overall frame shrink by 20-30 %.