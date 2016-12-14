#![feature(test)]
#![feature(custom_derive)]
#![feature(stmt_expr_attributes)]
#![feature(conservative_impl_trait)]

#![recursion_limit = "1024"]

#[macro_use]


extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;
extern crate num;
extern crate chrono;
extern crate error_chain;
extern crate itertools;


/// # Utah
///
/// ## Table of contents
///
/// + [DataFrame](#dataframe)
/// + [Transformation](#transformation)
/// + [Aggregation](#aggregation)
/// + [Imputation](#imputation)

///
/// ## DataFrame
/// Utah is a dataframe crate for Rust.
/// ### What's a dataframe?
/// ### Why use this crate?

/// ## Transformation
///

/// ## Aggregation
///

/// ## Imputation

pub mod dataframe;
mod tests;
pub mod error;
mod from;
pub mod types;
pub mod aggregate;
pub mod transform;
pub mod traits;
pub mod impute; 
