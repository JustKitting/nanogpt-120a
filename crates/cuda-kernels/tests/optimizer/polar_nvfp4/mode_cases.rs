#[path = "mode_cases/gram.rs"]
mod gram;
#[path = "mode_cases/gram_form.rs"]
mod gram_form;

pub(super) use gram::gram_correction_modes;
pub(super) use gram_form::gram_form_correction_modes;
