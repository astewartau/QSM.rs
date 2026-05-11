//! Combined pipeline algorithms
//!
//! Algorithms that span multiple QSM pipeline stages (e.g., combined
//! phase unwrapping and background field removal).

pub mod iharperella;

pub use iharperella::{iharperella, iharperella_default, iharperella_with_progress, IharperellaParams};
