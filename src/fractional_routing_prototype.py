def surface_height(hex: tuple[float, float]) -> float:
    return hex[0] + hex[1]

"""
Source is a tuple of (height, fluid)

Neighbors is a list of six tuples of (height, fluid)

Output is a list of one float for each neighbor

An implementation in another language might hard-code having 6 neighbors but not necessary here.
"""
def route_water(source, neighbors) -> list[float]:
    diffs_with_index = [(surface_height(neighbor) - surface_height(source), i) for i, neighbor in enumerate(neighbors)]
    diffs_with_index.sort(key=lambda x: x[0])
    
    result = [0.0] * len(neighbors)
    fluid_to_route = source[1]

    for j in range(len(diffs_with_index)):
        # Might not technically need this but a safety check
        if fluid_to_route <= 0.0:
            break

        diff, i = diffs_with_index[j]

        if surface_height(source) <= surface_height(neighbors[i]):
            break

        if fluid_to_route <= diff * -1:
            result[i] = fluid_to_route
            fluid_to_route = 0.0
        else:
            result[i] = diff * -1
            fluid_to_route -= diff * -1

    return result

# Only routing 90% of the fluid is a compromise between (1) quickly getting fluid out of the source in the case where fluid is coming into the
# source from elsewhere and (2) allowing levels to eventually equalize so fluid doesn't slosh back and forth between two hexes endlessly. In the
# main program this is the FLOW_FACTOR constant.
print(route_water((10.0, 10.0), [(10.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
print(route_water((10.0, 10.0), [(0.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Intermediate case, as dictated by the existing code:
print(route_water((10.0, 10.0), [(5.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [9.5, 0.0, 0.0, 0.0, 0.0, 0.0]
# Two equally low neighbors results in flow getting distributed evenly.
print(route_water((10.0, 10.0), [(10.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [4.5, 4.5, 0.0, 0.0, 0.0, 0.0]
# Uneven neighbors results in neighbors getting their surface level equalized.
print(route_water((10.0, 10.0), [(10.0, 0.0), (12.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [5.5, 3.5, 0.0, 0.0, 0.0, 0.0]
# A case to illustrate what seems to me to be an intuitive way to have FLOW_FACTOR interact with fractional routing.
print(route_water((10.0, 10.0), [(5.0, 0.0), (5.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0), (20.0, 0.0)])) # [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]