
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
use traits::{Transform, DataframeOps, RawDataframeConstructor, Aggregate, ToDataFrame};

fn main() {
    let a = arr2(&[[2., 7.], [3., NAN], [2., 4.]]);
    let b = arr2(&[[2., 6.], [3., 4.]]);
    let c = arr2(&[[2., 6.], [3., 4.], [2., 1.]]);
    let mut df: DataFrame<f64, String> =
        DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2", "3"]).unwrap();
    let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2"]).unwrap();
    let new_data = c.column(1).mapv(InnerType::from);
    let remove_idx = vec!["1"];
    let select_idx = vec!["2"];
    let append_idx = "8";

    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    df.impute(ImputeStrategy::Mean, UtahAxis::Column).to_df();
    let j = df.df_iter(UtahAxis::Row)
        .remove(&remove_idx[..])
        .select(&select_idx[..])
        .append(&append_idx, new_data.view())
        .sumdf()
        .to_df();
    println!("{:?}", j);
    let res: DataFrame<f64, String> = df.impute(ImputeStrategy::Mean, UtahAxis::Column)
        .to_df();
    // .to_df();    // res.mapdf(|x| x.as_ref())
    println!("{:?}", res);
    // df.impute(ImputeStrategy::Mean, UtahAxis::Column).to_df();
    let res_1: DataFrame<InnerType, OuterType> = df.inner_left_join(&df_1).to_df();
    println!("join result - {:?}", res_1);

}
