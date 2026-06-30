use crate::sweep::config::SweepConfig;

#[derive(Clone, Copy)]
pub(super) enum Stage {
    Screen,
    Full,
}

impl Stage {
    pub(super) fn event_prefix(self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Full => "training",
        }
    }

    pub(super) fn log_name(self) -> &'static str {
        match self {
            Self::Screen => "screen.log",
            Self::Full => "train.log",
        }
    }

    pub(super) fn max_seconds(self, config: &SweepConfig) -> f64 {
        match self {
            Self::Screen => config.screen_max_seconds,
            Self::Full => config.max_seconds,
        }
    }
}
