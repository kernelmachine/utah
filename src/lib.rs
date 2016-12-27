#![feature(test)]
#![feature(custom_derive)]
#![feature(stmt_expr_attributes)]
#![feature(conservative_impl_trait)]
#![feature(specialization)]
#![recursion_limit = "1024"]

#[macro_use]

extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;
extern crate num;
extern crate chrono;
#[macro_use]
extern crate error_chain;
extern crate itertools;
extern crate rustc_serialize;
extern crate csv;




pub mod combinators;
pub mod dataframe;
mod tests;
#[macro_use]
pub mod util;
mod implement;
pub mod mixedtypes;
mod bench;
pub mod prelude;
