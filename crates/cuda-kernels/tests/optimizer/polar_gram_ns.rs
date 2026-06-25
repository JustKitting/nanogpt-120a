use std::error::Error;

#[path = "polar_gram_ns/device.rs"]
mod device;
#[path = "polar_gram_ns/reference.rs"]
mod reference;

use crate::polar_vector;

#[test]
fn stabilized_gram_ns_matches_standard_update_direction() {
    for (rows, cols) in [(32, 64), (64, 256), (128, 512)] {
        let grad = reference::gradient(rows, cols);
        let standard = polar_vector::first_iteration_update(&grad, rows, cols, 0.95, 0.01, 0.1, 5);
        let gram_ns = reference::first_iteration_update(&grad, rows, cols, 0.95, 0.01, 0.1, 5);

        let cosine = reference::cosine(&standard, &gram_ns);
        let rel_l2 = reference::relative_l2(&gram_ns, &standard);
        assert!(
            cosine >= 0.999 && rel_l2 <= 5.0e-2,
            "rows={rows} cols={cols} cosine={cosine:.8} rel_l2={rel_l2:.8e}",
        );
    }
}

#[test]
fn stabilized_gram_ns_has_lower_rectangular_product_count() {
    let standard = reference::standard_cost(5, 4);
    let gram_ns = reference::stabilized_gram_ns_cost(5, 4, &[2]);

    assert!(gram_ns.rectangular_products < standard.rectangular_products);
    assert!(gram_ns.weighted_products < standard.weighted_products);
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn device_stabilized_gram_ns_reports_timing() -> Result<(), Box<dyn Error>> {
    device::run_timing_case(128, 512)
}
