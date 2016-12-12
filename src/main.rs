
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
pub mod from;
pub mod types;

use ndarray::{arr2, Axis};
use dataframe::*;
use types::*;
use dataframe::DFIter;

fn main() {
    let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    let b = arr2(&[[2, 6], [3, 4]]);
    let c = arr2(&[[2, 6], [3, 4]]);
    let df = DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2"]).unwrap();
    let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2"]).unwrap();
    let new_data = c.row(1).mapv(InnerType::from);
    let remove_idx = vec!["1"];
    let select_idx = vec!["2"];
    let append_idx = "1";

    let df_iter: DataFrameIterator = df_1.df_iter(Axis(0));
    let j: Append<Select<Remove<DataFrameIterator>>> = df.df_iter(Axis(0))
        .remove(&remove_idx[..])
        .select(&select_idx[..])
        .append(&append_idx, new_data.view());
    let res: InnerJoin<Append<Select<Remove<DataFrameIterator>>>> = j.inner_left_join(df_iter);

    let res: Vec<_> = df.df_iter(Axis(1))
        .sumdf()
        .collect();
    println!("{:?}", res);


}
