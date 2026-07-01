use super::eval::ScheduleResult;
use super::schedule_name;

pub(super) fn report_schedule(result: &ScheduleResult) {
    report_schedule_result("nvfp4_gram_form_schedule", result);
}

pub(in super::super) fn report_schedule_search_best(kind: &str, result: &ScheduleResult) {
    report_schedule_result(
        format_args!("nvfp4_gram_form_schedule_search_best kind={kind}"),
        result,
    );
}

fn report_schedule_result(label: impl std::fmt::Display, result: &ScheduleResult) {
    println!(
        "{label} name={} finite={} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={} schedule={}",
        result.name,
        result.finite,
        result.cosine,
        result.rel_l2,
        result.max_abs,
        result.nvfp4_grams,
        result.hi_grams,
        result.rejected,
        schedule_name(&result.schedule),
    );
}
