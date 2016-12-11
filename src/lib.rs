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

pub mod dataframe;
pub mod tests;
pub mod error;
pub mod from;
pub mod types;
