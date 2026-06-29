#[path = "../sweep/mod.rs"]
mod sweep;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    sweep::run()
}
