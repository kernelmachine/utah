use ndarray::{arr2, arr1, stack};
use dataframe::*;
use test::Bencher;
use ndarray::Array;
use rand::distributions::Range;
use ndarray_rand::RandomExt;
use std::rc::Rc;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, BTreeMap};
use util::types::*;
use util::error::*;
use combinators::aggregate::*;
use combinators::transform::*;
use combinators::interact::*;
use util::traits::*;
use std::f64::NAN;
use ndarray::Axis;
use mixedtypes::*;


#[bench]
fn bench_creation(b: &mut Bencher) {
    let a = Array::random((10, 5), Range::new(0., 10.));
    b.iter(|| {
        let _: Result<DataFrame<f64, String>> = DataFrame::new(a.clone())
            .columns(&["1", "2", "3", "4", "5"]);
    });
}
