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
extern crate error_chain;
extern crate itertools;

macro_rules! dataframe {
 {
    $ (
        $column:item : $data:item
    ),
    *
} => {
    let mut c = Vec::new();
    let mut n = Vec::new();

    $(
        c.extend($data.iter().map(|x| x.to_owned()));
        n.push($column.to_owned())
        let row_len = $data.iter().fold(0, |acc, _| acc + 1);
    )*
    let res_dim = (row_len, n.len());
    DataFrame {
        columns: n,
        data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
        index: (0..),
    }

}
}
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
pub mod process;
pub mod join;
