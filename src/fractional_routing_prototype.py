def surface_height(hex: tuple[float, float]) -> float:
    return hex[0] + hex[1]

FLOW_FACTOR = 0.90

TEST_CASES = [
    [(10.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [(0.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [(5.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [9.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    [(10.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [4.5, 4.5, 0.0, 0.0, 0.0, 0.0]
    [(10.0, 0.0), (12.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [5.5, 3.5, 0.0, 0.0, 0.0, 0.0]
    [(5.0, 0.0), (5.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)], # [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    [(10.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 0.0)], # [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)], # [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
]

def distribute_fluid(fluid: float, targets: list[float]) -> list[float]:
    result = [0.0] * len(targets)

    targets_with_indices = list(enumerate(targets))
    targets_with_indices.sort(key=lambda x: x[1])

    for j in range(len(targets)):
        target = targets_with_indices[j][1]
        next_neighbor = targets_with_indices[j + 1][1] if j + 1 < len(targets) else float('inf')

        fluid_to_distribute = min((next_neighbor - target) * (j + 1), fluid)

        for i in range(j + 1):
            result[i] += fluid_to_distribute / (j + 1)

        fluid -= fluid_to_distribute
        if fluid == 0.0:
            break

    return result

print("# Distribute fluid")
print (distribute_fluid(9.0, [surface_height(hex) for hex in TEST_CASES[0]]))
print (distribute_fluid(10.0, [surface_height(hex) for hex in TEST_CASES[1]]))
print (distribute_fluid(9.5, [surface_height(hex) for hex in TEST_CASES[2]]))
print (distribute_fluid(9.0, [surface_height(hex) for hex in TEST_CASES[3]]))
print (distribute_fluid(9.0, [surface_height(hex) for hex in TEST_CASES[4]]))
print (distribute_fluid(10.0, [surface_height(hex) for hex in TEST_CASES[5]]))

def fluid_to_distribute(source: tuple[float, float], neighbors: list[tuple[float, float]]) -> float:
    result = 0.0

    fluid_to_distribute = source[1]

    gravity_flows = []
    for neighbor in neighbors:
        if fluid_to_distribute <= 0.0:
            break

        first_part = min(fluid_to_distribute, max(0, source[0] - surface_height(neighbor)))
        gravity_flows.append(first_part)

        result += first_part
        fluid_to_distribute -= first_part

    for i, neighbor in enumerate(neighbors):
        if fluid_to_distribute <= 0.0:
            break

        second_part = max(0, min(source[1], (surface_height(source) - surface_height(neighbor))) - gravity_flows[i])
        second_part = min(second_part, fluid_to_distribute)
        fluid_to_distribute -= second_part
        result += second_part * FLOW_FACTOR

    return min(source[1], result)

print("# Fluid to distribute")
for case in TEST_CASES[3:]:
    print(fluid_to_distribute((10.0, 10.0), case))

"""
Source is a tuple of (height, fluid)

Neighbors is a list of six tuples of (height, fluid)

Output is a list of one float for each neighbor

An implementation in another language might hard-code having 6 neighbors but not necessary here.
"""
def route_water(source: tuple[float, float], neighbors: list[tuple[float, float]]) -> list[float]:
    # TODO: Math to figure out how much fluid to distribute in total.
    
    fluid = fluid_to_distribute(source, neighbors)

    return distribute_fluid(fluid, [surface_height(neighbor) for neighbor in neighbors])

print("# Route water")
# Only routing 90% of the fluid is a compromise between (1) quickly getting fluid out of the source in the case where fluid is coming into the
# source from elsewhere and (2) allowing levels to eventually equalize so fluid doesn't slosh back and forth between two hexes endlessly. In the
# main program this is the FLOW_FACTOR constant.
print(route_water((10.0, 10.0), TEST_CASES[0])) # [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# But in this scenario we can move all the fluid because there's no chance of it sloshing back. Relevant existing code:
"""
        var move_f = 0.0;
        if (2.0 * f <= diff) {
            move_f = f;
        } else if (f < diff && diff < 2.0 * f) {
            move_f = (diff - f) + (2.0 * f - diff) * constants.flow_factor;
        } else { // diff <= f
            move_f = diff * constants.flow_factor;
        }
"""
print(route_water((10.0, 10.0), TEST_CASES[1])) # [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Intermediate case
print(route_water((10.0, 10.0), TEST_CASES[2])) # [9.5, 0.0, 0.0, 0.0, 0.0, 0.0]
# Two equally low neighbors results in flow getting distributed evenly.
print(route_water((10.0, 10.0), TEST_CASES[3])) # [4.5, 4.5, 0.0, 0.0, 0.0, 0.0]
# Uneven neighbors results in neighbors getting their surface level equalized.
print(route_water((10.0, 10.0), TEST_CASES[4])) # [5.5, 3.5, 0.0, 0.0, 0.0, 0.0]
# A case to illustrate what seems to me to be an intuitive way to have FLOW_FACTOR interact with fractional routing.
print(route_water((10.0, 10.0), TEST_CASES[5])) # [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
# All target hexes are the same height so fluid is distributed evenly.
print(route_water((10.0, 10.0), TEST_CASES[6])) # [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
# As previous case but with water instead of land—difference should not matter.
print(route_water((10.0, 10.0), TEST_CASES[7])) # [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]