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