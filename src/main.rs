
#![feature(test)]
extern crate ndarray;
extern crate test;
extern crate rand;
extern crate ndarray_rand;



use ndarray::arr2;
mod dataframe;
fn main() {
    let a = arr2(&[[2., 3.], [3., 4.], [7., 34.]]);
    let names = vec!["a", "b"];
    let names1 = vec!["a", "c"];

    let b = arr2(&[[2., 3.], [7., 8.]]);
    let df = dataframe::DataFrame::from_array(&a, &names).unwrap();
    let df1 = dataframe::DataFrame::from_array(&b, &names1).unwrap();
    let j = df.inner_join(&df1, "a");
}
