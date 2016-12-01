#![feature(test)]
#![feature(custom_derive)]
#![feature(stmt_expr_attributes)]
#![recursion_limit = "1024"]

#[macro_use]


extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;
extern crate num;
extern crate chrono;
extern crate error_chain;


pub mod dataframe;
pub mod tests;
pub mod helper;
pub mod join;
pub mod error;
