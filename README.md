I created this after experimenting with existing erosion simulatiion software and being dissatisfied with features like not being able to handle connecting to off-map oceans and issues like doing erosion without deposition or not handling river deltas correctly for other reasons.

In addition to having river inflows/outflows, I experimented with having an off-map river source, but found it difficult to get sediment inflows right, and found it more effective to have an admittedly highly artificial river basin on the east end of the map.

Below reading list is mostly things I haven't incorporated but may in the future.

To run a quick (~5 minute) test:
```
cargo run --release -- --rounds 10
```

Another example usage:
```
cargo run --release -- --rounds 10000 --seed 4064140151
```

# Multi-Stage Simulation:

Simulating ~20,000 years in three stages:
* 10,000 rounds at 7 steps per day: ~17342 years
* 10,000 rounds at (24*2) steps per day: ~2529 years
* 10,000 rounds at (24*60) steps per day: ~84 years


Reading list
------------
• [Liang et al. 2015](https://esurf.copernicus.org/articles/3/67/2015/esurf-3-67-2015.html) – DeltaRCM reduced-complexity delta model ([GitHub](https://github.com/DeltaRCM/pyDeltaRCM)).
• Caldwell & Edmonds 2014, "The effects of sediment properties on deltaic processes and morphologies: A numerical modeling
study" and Edmonds & Slingerland 2010, "Significant effect of sediment cohesion on delta morphology" – grain-size & cohesion effects.
• [Nardin & Fagherazzi 2012](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012GL051788) – wave influence on mouth-bar evolution.
• Julien 2010, Erosion and Sedimentation, 2nd ed – sediment transport formulas for rivers.