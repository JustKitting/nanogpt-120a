#[derive(Clone, Copy)]
pub(in super::super) struct GramRequest<'s> {
    pub(in super::super) source: &'s [f32],
    pub(in super::super) rows: usize,
    pub(in super::super) cols: usize,
    pub(in super::super) iter: usize,
}

impl<'s> GramRequest<'s> {
    pub(in super::super) fn new(source: &'s [f32], rows: usize, cols: usize, iter: usize) -> Self {
        Self {
            source,
            rows,
            cols,
            iter,
        }
    }
}

pub(in super::super) struct CorrectionGram {
    pub(in super::super) values: Vec<f32>,
    pub(in super::super) stale_reject_candidate: bool,
    pub(in super::super) refresh: bool,
}

impl CorrectionGram {
    pub(super) fn refreshed(values: Vec<f32>) -> Self {
        Self {
            values,
            stale_reject_candidate: false,
            refresh: true,
        }
    }

    pub(super) fn approximate(values: Vec<f32>) -> Self {
        Self {
            values,
            stale_reject_candidate: false,
            refresh: false,
        }
    }
}
