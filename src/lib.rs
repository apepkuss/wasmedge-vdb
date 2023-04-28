#[macro_use]
extern crate num_derive;

pub mod client;
pub mod common;
pub mod error;
pub mod proto;
pub mod schema;
pub mod utils;

pub const WAIT_LOAD_DURATION_MS: u64 = 500;
