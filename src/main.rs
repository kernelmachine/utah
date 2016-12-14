
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
pub mod impute;

use ndarray::arr2;
use dataframe::*;
use types::*;
use traits::DFIter;
// use transform::*;

fn main() {
    let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    // let b = arr2(&[[2, 6], [3, 4]]);
    // let c = arr2(&[[2, 6], [3, 4]]);
    let df = DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2"]).unwrap();
    // let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2"]).unwrap();
    // let new_data = c.row(1).mapv(InnerType::from);
    // let remove_idx = vec!["1"];
    // let select_idx = vec!["2"];
    // let append_idx = "1";

    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    // let j: Append<Select<Remove<DataFrameIterator>>> = df.df_iter(UtahAxis::Row)
    //     .remove(&remove_idx[..])
    //     .select(&select_idx[..])
    //     .append(&append_idx, new_data.view());

    let res: Vec<_> = df.df_iter(UtahAxis::Column)
        .sumdf()
        .collect();
    println!("{:?}", res);


}
