
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
extern crate rustc_serialize;
extern crate csv;


pub mod adapters;
pub mod dataframe;
pub mod tests;
#[macro_use]
pub mod util;
pub mod implement;
pub mod mixedtypes;
pub mod bench;

use ndarray::{arr1, arr2};
use dataframe::*;
use util::types::*;

use std::f64::NAN;
use util::traits::*;
use util::error::*;
use ndarray::{Axis, ArrayView};
use ndarray::stack;
use util::readcsv::*;
use mixedtypes::*;
use std::collections::BTreeMap;

fn main() {
    let mut btree: BTreeMap<String, Row<f64>> = BTreeMap::new();
    btree.insert("a".to_string(), arr1(&[2., 3., 2.]));


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

    let df: Result<DataFrame<InnerType, OuterType>> = DataFrame::read_csv("/Users/suchin/Github/rust-dataframe/src/tests/test.\
                                                                           csv");
    let a = arr2(&[[2., 7.], [3., NAN], [2., 4.]]);
    // let b = arr2(&[[2., 6.], [3., 4.]]);
    let c = arr2(&[[2., 6.], [3., 4.], [2., 1.]]);
    let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"])?
        .index(&["1", "2", "3"])?;
    let df_1 = DataFrame::new(c).columns(&["c", "d"])?.index(&["1", "2", "3"])?;
    let new_data = df.select(&["2"], UtahAxis::Row).as_array()?;

    // let df_iter: DataFrameIterator = df_1.df_iter(UtahAxis::Row);
    let j = df.df_iter(UtahAxis::Row)
        .remove(&["1"])
        .select(&["2"])
        .append("8", new_data.view())
        .sumdf()
        .as_df()?;
    let res: DataFrame<f64, String> = df.impute(ImputeStrategy::Mean, UtahAxis::Column)
        .as_df()?;

    let res_1: DataFrame<f64, String> = df.inner_left_join(&df_1).as_df()?;
    let concat = df.concat(&df_1, UtahAxis::Row).as_df();
    // let b = arr1(&[2., 3., 2.]);
    let k: DataFrame<f64, String> = dataframe!(
    {
        "a" =>  col!([2., 3., 2.]),
        "b" =>  col!([2., NAN, 2.])
    });
    println!("{:?}", k);
    Ok(())

}
