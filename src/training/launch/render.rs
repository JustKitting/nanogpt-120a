use std::io::IsTerminal;

mod boxed;

use burn::train::Interrupter;
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{CliMetricsRenderer, MetricsRenderer};

pub(super) use boxed::BoxedMetricsRenderer;

pub(super) fn default_renderer(interrupter: Interrupter) -> Box<dyn MetricsRenderer> {
    let mode = super::env_nonempty("TRAIN_RENDERER").unwrap_or_else(|| "auto".to_string());
    let persistent = matches!(mode.as_str(), "tui-persistent" | "persistent")
        || super::env_bool("TRAIN_RENDERER_PERSIST").unwrap_or(false);
    let wants_tui = matches!(
        mode.as_str(),
        "auto" | "tui" | "tui-persistent" | "persistent"
    );

    if wants_tui && std::io::stdout().is_terminal() {
        let renderer = TuiMetricsRendererWrapper::new(interrupter, None);
        return if persistent {
            Box::new(renderer.persistent())
        } else {
            Box::new(renderer)
        };
    }

    if matches!(mode.as_str(), "tui" | "tui-persistent" | "persistent") {
        eprintln!("train_renderer_fallback=cli reason=stdout_not_tty requested={mode}");
    }
    Box::new(CliMetricsRenderer::new())
}
