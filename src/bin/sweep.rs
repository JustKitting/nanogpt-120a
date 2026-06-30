#[path = "../fs_utils.rs"]
mod fs_utils;
#[path = "../sweep/mod.rs"]
mod sweep;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    sweep::run()
}
