mod checkpoint;
mod fs_utils;
mod time_utils;
mod training;
mod upload;

use std::error::Error;

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

fn main() -> AppResult {
    training::launch_from_env()
}
