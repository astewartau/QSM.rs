//! QSM-Core: Quantitative Susceptibility Mapping algorithms
//!
//! This crate provides QSM algorithms for medical image processing.
//!
//! # Modules
//! - `fft`: 3D FFT operations using rustfft
//! - `kernels`: Dipole, SMV, and Laplacian kernels
//! - `unwrap`: Phase unwrapping (ROMEO, Laplacian)
//! - `bgremove`: Background field removal (SHARP, V-SHARP, PDF, iSMV)
//! - `inversion`: Dipole inversion (TKD, Tikhonov, TV, RTS, MEDI)
//! - `solvers`: Iterative solvers (CG, LSMR)
//! - `utils`: Gradient operators, padding, etc.
//! - `bet`: Brain extraction tool

// Core modules
pub mod fft;
pub mod priority_queue;
pub mod region_grow;

// Algorithm modules
pub mod kernels;
pub mod unwrap;
pub mod bgremove;
pub mod inversion;
pub mod solvers;
pub mod utils;

// I/O modules
pub mod nifti_io;

// Brain extraction
pub mod bet;
