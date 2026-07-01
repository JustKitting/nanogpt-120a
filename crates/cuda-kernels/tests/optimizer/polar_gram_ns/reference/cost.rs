pub fn standard_cost(iterations: usize, aspect_ratio: usize) -> ProductCost {
    ProductCost {
        rectangular_products: iterations * 2,
        weighted_products: iterations * (2 * aspect_ratio + 1),
    }
}

pub fn stabilized_gram_ns_cost(
    iterations: usize,
    aspect_ratio: usize,
    resets: &[usize],
) -> ProductCost {
    let restart_count = resets
        .iter()
        .filter(|&&iter| iter != 0 && iter < iterations)
        .count();
    let rectangular_products = 1 + restart_count * 2 + 1;
    let weighted_products = 4 * iterations + 6 * aspect_ratio * restart_count - 6 * restart_count;

    ProductCost {
        rectangular_products,
        weighted_products,
    }
}

#[derive(Clone, Copy)]
pub struct ProductCost {
    pub rectangular_products: usize,
    pub weighted_products: usize,
}
