
#![feature(test)]
#[macro_use]

extern crate ndarray;
extern crate test;
extern crate rand;
extern crate ndarray_rand;
extern crate num;
extern crate chrono;
extern crate error_chain;

pub mod dataframe;
pub mod helper;
pub mod join;
pub mod error;
pub mod from;
pub mod types;

fn main() {


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
