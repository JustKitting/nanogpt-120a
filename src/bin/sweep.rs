#[path = "../fs_utils.rs"]
mod fs_utils;
#[path = "../sweep/mod.rs"]
mod sweep;
#[path = "../time_utils.rs"]
mod time_utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    sweep::run()
}
