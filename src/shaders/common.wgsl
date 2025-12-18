fn total_elevation(cell: Hex) -> f32 {
    return cell.elevation + cell.elevation_residual;
}

fn total_water_depth(cell: Hex) -> f32 {
    return max(cell.water_depth + cell.water_depth_residual, 0.0);
}

fn height(cell: Hex) -> f32 {
    return total_elevation(cell) + total_water_depth(cell) + cell.suspended_load;
}

fn total_fluid(cell: Hex) -> f32 {
    return total_water_depth(cell) + cell.suspended_load;
}

fn sediment_fraction(cell: Hex) -> f32 {
    let tf = total_fluid(cell);
    if (tf <= 0.0) {
        return 0.0;
    }
    return cell.suspended_load / tf;
}