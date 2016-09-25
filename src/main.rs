
#[macro_use(stack)]
extern crate ndarray;
extern crate test;
extern crate rand;
extern crate ndarray_rand;

use test::Bencher;
use rand::distributions::Range;
use ndarray_rand::RandomExt;

use ndarray::arr2;
mod lib;
fn main() {
    let a = arr2(&[[2., 3.], [3., 4.]]);
    let names = vec!["a", "b"];
    let df = lib::DataFrame::from_array(&a, &names);
}
