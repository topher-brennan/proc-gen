fn height(cell: Hex) -> f32 {
    return cell.elevation + cell.water_depth + cell.suspended_load;
}

fn total_fluid(cell: Hex) -> f32 {
    return cell.water_depth + cell.suspended_load;
}

fn sediment_fraction(cell: Hex) -> f32 {
    return cell.suspended_load / total_fluid(cell);
}