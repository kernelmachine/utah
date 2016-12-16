
#![feature(test)]
#![feature(conservative_impl_trait)]

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
use traits::{Transform, Process, Join};

fn main() {
    let a = arr2(&[[2.0, 7.0], [3.0, NAN], [2.0, 4.0]]);
    let b = arr2(&[[2, 6], [3, 4]]);
    let c = arr2(&[[2.0, 6.0], [3.0, 4.0], [2.0, NAN]]);
    let mut df = DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2", "3"]).unwrap();
    let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2"]).unwrap();
    let new_data = c.column(1).mapv(InnerType::from);
    let remove_idx = vec!["1"];
    let select_idx = vec!["2"];
    let append_idx = "8";

    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    let mut j = df.df_iter(UtahAxis::Column)
        .remove(&remove_idx[..])
        .select(&select_idx[..])
        .append(&append_idx, new_data.view())
        .to_df();
    println!("{:?}", j);
    // let res: DataFrame = df.impute(ImputeStrategy::Mean, UtahAxis::Column).to_df();    // res.mapdf(|x| x.as_ref())
    // println!("{:?}", res);
    df.impute(ImputeStrategy::Mean, UtahAxis::Column).to_df();
    let res_1: DataFrame = df.inner_left_join(&df_1).to_df();
    println!("join result - {:?}", res_1);

}
