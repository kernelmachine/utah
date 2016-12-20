
#![feature(test)]
#![feature(conservative_impl_trait)]
#![feature(specialization)]

#[macro_use]

extern crate ndarray;
extern crate test;
extern crate rand;
extern crate ndarray_rand;
extern crate num;
extern crate chrono;
extern crate error_chain;
extern crate itertools;

pub mod dataframe;
pub mod error;
mod from;
pub mod types;
pub mod read;
pub mod traits;
pub mod transform;
pub mod aggregate;
pub mod process;
pub mod join;

use ndarray::arr2;
use dataframe::*;
use types::*;

use std::f64::NAN;
use traits::{Transform, Operations, Constructor, Aggregate, ToDataFrame};

fn main() {
    let a = arr2(&[[2., 7.], [3., NAN], [2., 4.]]);
    let b = arr2(&[[2., 6.], [3., 4.]]);
    let c = arr2(&[[2., 6.], [3., 4.], [2., 1.]]);
    let mut df: DataFrame<f64, String> =
        DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2", "3"]).unwrap();
    let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2", "3"]).unwrap();
    let new_data = df.select(&["2"], UtahAxis::Row).as_array();

    println!("{:?}", new_data);
    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    df.impute(ImputeStrategy::Mean, UtahAxis::Column).as_df();
    let j = df.df_iter(UtahAxis::Row)
        .remove(&["1"])
        .select(&["2"])
        .append("8", new_data.view())
        .as_df();
    println!("{:?}", j);
    let res: DataFrame<f64, String> = df.impute(ImputeStrategy::Mean, UtahAxis::Column)
        .as_df();

    println!("{:?}", res);
    let res_1: DataFrame<f64, String> = df.inner_left_join(&df_1).as_df();
    println!("join result - {:?}", res_1);

}
