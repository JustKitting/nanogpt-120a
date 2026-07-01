use super::{COLS, MAX_ITERATIONS, ROWS, TestResult, math, with_polar};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_rht_polar_estimator_reports_update_error() -> TestResult {
    with_polar(|polar| {
        let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
        let gram_expected = math::matmul_f16_leaf(&source, &source, ROWS, ROWS, COLS);
        let gram_actual = polar.product(&source, &source, ROWS, ROWS, COLS, 0, 0)?;
        let (cosine, rel_l2, max_abs) = math::error_metrics(&gram_actual, &gram_expected);
        println!(
            "nvfp4_rht_polar first_gram cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
        );
        let source_t = math::transpose(&source, ROWS, COLS);
        let ax_expected = math::matmul_f16_leaf(&gram_expected, &source_t, ROWS, COLS, ROWS);
        let ax_actual = polar.product(&gram_expected, &source_t, ROWS, COLS, ROWS, 0, 1)?;
        let (cosine, rel_l2, max_abs) = math::error_metrics(&ax_actual, &ax_expected);
        println!(
            "nvfp4_rht_polar first_ax_from_expected_gram cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
        );
        let ax_t = math::transpose(&ax_expected, ROWS, COLS);
        let aax_expected = math::matmul_f16_leaf(&gram_expected, &ax_t, ROWS, COLS, ROWS);
        let aax_actual = polar.product(&gram_expected, &ax_t, ROWS, COLS, ROWS, 0, 2)?;
        let (cosine, rel_l2, max_abs) = math::error_metrics(&aax_actual, &aax_expected);
        println!(
            "nvfp4_rht_polar first_aax_from_expected_inputs cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
        );

        for iterations in 1..=MAX_ITERATIONS {
            let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
            let actual = polar.iterations(source.clone(), ROWS, COLS, iterations)?;
            let (cosine, rel_l2, max_abs) = math::error_metrics(&actual, &expected);
            println!(
                "nvfp4_rht_polar iterations={iterations} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
            );
        }

        let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, 5);
        for fp4_prefix in 1..=3 {
            let mut actual = source.clone();
            for iter in 0..5 {
                actual = if iter < fp4_prefix {
                    polar.step(&actual, ROWS, COLS, iter)?
                } else {
                    math::polar_step_f16_leaf(&actual, ROWS, COLS, iter)
                };
            }
            let (cosine, rel_l2, max_abs) = math::error_metrics(&actual, &expected);
            println!(
                "nvfp4_rht_polar hybrid_fp4_prefix={fp4_prefix} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
            );
        }

        Ok(())
    })
}
