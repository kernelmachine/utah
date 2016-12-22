
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

#[macro_use]
pub mod macros;

use ndarray::arr2;
use dataframe::*;
use types::*;

use std::f64::NAN;
use traits::{Transform, Operations, Constructor, Aggregate, ToDataFrame};
use error::*;
use ndarray::{Axis, ArrayView};
use ndarray::stack;

fn main() {
    if let Err(ref e) = run() {
        use ::std::io::Write;
        let stderr = &mut ::std::io::stderr();
        let errmsg = "Error writing to stderr";

        writeln!(stderr, "error: {}", e).expect(errmsg);

        for e in e.iter().skip(1) {
            writeln!(stderr, "caused by: {}", e).expect(errmsg);
        }

        // The backtrace is not always generated. Try to run this example
        // with `RUST_BACKTRACE=1`.
        if let Some(backtrace) = e.backtrace() {
            writeln!(stderr, "backtrace: {:?}", backtrace).expect(errmsg);
        }

        ::std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let a = arr2(&[[2., 7.], [3., NAN], [2., 4.]]);
    // let b = arr2(&[[2., 6.], [3., 4.]]);
    let c = arr2(&[[2., 6.], [3., 4.], [2., 1.]]);
    let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"])?
        .index(&["1", "2", "3"])?;
    let df_1 = DataFrame::new(c).columns(&["c", "d"]).unwrap().index(&["1", "2", "3"]).unwrap();
    let new_data = df.select(&["2"], UtahAxis::Row).as_array();

    println!("{:?}", new_data);
    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    df.impute(ImputeStrategy::Mean, UtahAxis::Column).as_df();
    let j = df.df_iter(UtahAxis::Row)
        .remove(&["1"])
        .select(&["2"])
        .append("8", new_data.view())
        .sumdf()
        .as_df();
    println!("{:?}", j);
    let res: DataFrame<f64, String> = df.impute(ImputeStrategy::Mean, UtahAxis::Column)
        .as_df();

    println!("{:?}", res);
    let res_1: DataFrame<f64, String> = df.inner_left_join(&df_1).as_df();
    println!("join result - {:?}", res_1);
    let concat = df.concat(&df_1, UtahAxis::Row).as_df();
    println!("concat result - {:?}", concat);
    // let b = arr1(&[2., 3., 2.]);
    let k: DataFrame<f64, String> = dataframe!(
    {
        "a" =>  column!([2., 3., 2.]),
        "b" =>  column!([2., NAN, 2.])
    });
    println!("{:?}", k);
    Ok(())

}
