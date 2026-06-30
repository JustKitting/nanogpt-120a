use super::high_level;

#[test]
fn hadamard_levels_are_balanced_per_factor() {
    let rows = 16;
    for factor in 0..8 {
        let highs = (0..rows).filter(|row| high_level(*row, factor)).count();
        assert_eq!(highs, rows / 2);
    }
}

#[test]
fn hadamard_factor_pairs_cover_all_two_level_cells() {
    let rows = 16;
    for left in 0..4 {
        for right in left + 1..4 {
            let mut cells = [0; 4];
            for row in 0..rows {
                let a = usize::from(high_level(row, left));
                let b = usize::from(high_level(row, right));
                cells[a * 2 + b] += 1;
            }
            assert_eq!(cells, [rows / 4; 4]);
        }
    }
}
