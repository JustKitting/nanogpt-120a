pub(super) fn high_level(row: usize, factor: usize) -> bool {
    ((row + 1) & (factor + 1)).count_ones() & 1 == 0
}
