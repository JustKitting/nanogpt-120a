#[inline(always)]
pub(super) fn better(value: f32, index: u32, best_value: f32, best_index: u32) -> bool {
    value > best_value || (value == best_value && index < best_index)
}
