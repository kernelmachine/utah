use ndarray::{arr2, arr1, stack};
use dataframe::*;
#[cfg(nightly)]
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
use util::traits::*;
use std::f64::NAN;
use ndarray::Axis;
use combinators::interact::*;
use mixedtypes::*;

#[bench]
fn test_sumdf_2_2(b: &mut Bencher) {
    let c = Array::random((2, 2), Range::new(0., 10.));
    let mut c_names: Vec<String> = vec![];
    for i in 0..2 {
        c_names.push(i.to_string());
    }

    let mut c_index: Vec<String> = vec![];
    for i in 0..2 {
        c_index.push(i.to_string());
    }


    let mut c_df: DataFrame<f64, String> = DataFrame::new(c)
        .columns(&c_names[..])
        .unwrap()
        .index(&c_index[..])
        .unwrap();

    b.iter(|| {
        let _ = c_df.sumdf(UtahAxis::Column).as_df();
    });
}

#[bench]
fn test_sumdf_10_10(b: &mut Bencher) {
    let c = Array::random((10, 10), Range::new(0., 10.));


    let mut c_names: Vec<String> = vec![];
    for i in 0..10 {
        c_names.push(i.to_string());
    }



    let mut c_index: Vec<String> = vec![];
    for i in 0..10 {
        c_index.push(i.to_string());
    }


    let mut c_df: DataFrame<f64, String> = DataFrame::new(c)
        .columns(&c_names[..])
        .unwrap()
        .index(&c_index[..])
        .unwrap();

    b.iter(|| {
        // let c = c.clone().sum(Axis(1));
        let _ = c_df.sumdf(UtahAxis::Column).as_df();
    });
}

#[bench]
fn test_sumdf_100_100(b: &mut Bencher) {
    let c = Array::random((100, 100), Range::new(0., 10.));


    let mut c_names: Vec<String> = vec![];
    for i in 0..100 {
        c_names.push(i.to_string());
    }



    let mut c_index: Vec<String> = vec![];
    for i in 0..100 {
        c_index.push(i.to_string());
    }



    let mut c_df: DataFrame<f64, String> = DataFrame::new(c)
        .columns(&c_names[..])
        .unwrap()
        .index(&c_index[..])
        .unwrap();

    b.iter(|| {
        let _ = c_df.sumdf(UtahAxis::Column).as_df();
    });
}

#[bench]
fn test_sumdf_1000_10(b: &mut Bencher) {
    let c = Array::random((1000, 10), Range::new(0., 10.));


    let mut c_names: Vec<String> = vec![];
    for i in 0..10 {
        c_names.push(i.to_string());
    }



    let mut c_index: Vec<String> = vec![];
    for i in 0..1000 {
        c_index.push(i.to_string());
    }



    let mut c_df: DataFrame<f64, String> = DataFrame::new(c)
        .columns(&c_names[..])
        .unwrap()
        .index(&c_index[..])
        .unwrap();

    b.iter(|| {
        let _ = c_df.sumdf(UtahAxis::Row).as_df();
    });
}
