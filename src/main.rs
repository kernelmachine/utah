
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
pub mod helper;
pub mod error;
pub mod from;
pub mod types;

use ndarray::{arr2, Axis};
use dataframe::*;
use types::*;
use dataframe::DFIter;
use std::iter::Chain;
fn main() {
    let a = arr2(&[[2, 7], [3, 4]]);
    let b = arr2(&[[2, 6], [3, 4]]);
    let c = arr2(&[[2, 6], [3, 4]]);
    let df = DataFrame::new(a).columns(&["a", "b"]).unwrap().index(&["1", "2"]).unwrap();
    let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap().index(&["1", "2"]).unwrap();
    let new_data = c.row(1).mapv(InnerType::from);

    let j: Chain<Append<Select<Remove<DataFrameIterator>>>, DataFrameIterator> = df.df_iter(Axis(1))
        .remove(vec![OuterType::Str("a".to_string())])
        .select(vec![OuterType::Str("b".to_string())])
        .append(OuterType::Str("c".to_string()), new_data.view())
        .concat(df_1.df_iter(Axis(1)));
    // let concat : Vec<_> = df.iter(Axis(1)).unwrap().concat(df_1.iter(Axis(1)).unwrap()).collect();
    // let r : Row<InnerType> =  c.row(1).mapv(InnerType::Int);
    // let added : Vec<_> = df.iter(Axis(0)).unwrap().add("z", r.view()).collect();
    // let removed : Vec<_> = df.iter(Axis(0)).unwrap().remove("2").collect();
    let d: Vec<_> = j.collect();
    println!("{:?}", d);
    // dataframe!()
    // let a = arr2(&[[2., 3.], [3., 4.], [7., 34.]]);
    // let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
    // let names1 = vec!["a", "c"].iter().map(|x| x.to_string()).collect();
    // let b = arr2(&[[2., 3.], [7., 8.]]);
    // if let Ok(df) = dataframe::DataFrame::from_array(&a, &names) {
    //     if let Ok(df1) = dataframe::DataFrame::from_array(&b, &names1) {
    //         let j = df.inner_join(&df1, "a");
    //         println!("{:?}", j);
    //     }
    // }

}
