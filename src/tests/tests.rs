
extern crate rand;
#[cfg(nightly)]
extern crate test;
#[cfg(nightly)]
use test::Bencher;
use ndarray::Array;
use rand::distributions::Range;
use ndarray_rand::RandomExt;
use std::rc::Rc;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, BTreeMap};
use std::f64::NAN;
use prelude::*;

#[test]
fn outer_left_join() {
    let a = arr2(&[[1.], [4.]]);
    let left: DataFrame<InnerType, OuterType> =
        DataFrame::new(a).index(&[1, 2]).unwrap().columns(&["a"]).unwrap();
    let b = arr2(&[[5.]]);
    let right: DataFrame<InnerType, OuterType> =
        DataFrame::new(b).index(&[1]).unwrap().columns(&["b"]).unwrap();
    let res = left.outer_left_join(&right).as_df();
    let expected_data = arr2(&[[InnerType::Float(1.), InnerType::Float(5.)],
                               [InnerType::Float(4.), InnerType::Empty]]);
    // let expected = DataFrame {
    //     index: vec![OuterType::Int32(1), OuterType::Int32(2)],
    //     data: expected_data,
    //     columns: vec![OuterType::Str(String::from("a")), OuterType::Str(String::from("b"))],
    // };
    let expected =
        DataFrame::new(expected_data).index(&[1, 2]).unwrap().columns(&["a", "b"]).unwrap();
    assert_eq!(res.unwrap(), expected);

}
#[test]
fn inner_join() {
    let a = arr2(&[[1], [2], [3]]);
    let left: DataFrame<i32, OuterType> =
        DataFrame::new(a).index(&[1, 2, 3]).unwrap().columns(&["a"]).unwrap();

    let b = arr2(&[[5], [6]]);
    let right = DataFrame::new(b).index(&[1, 3]).unwrap().columns(&["b"]).unwrap();
    let res = left.inner_left_join(&right).as_df();
    let expected_data = arr2(&[[1, 5], [3, 6]]);
    let expected: DataFrame<i32, OuterType> =
        DataFrame::new(expected_data).index(&[1, 3]).unwrap().columns(&["a", "b"]).unwrap();

    assert_eq!(res.unwrap(), expected);
}

#[test]
fn dataframe_creation() {
    let a = arr2(&[[2., 3.], [3., 4.]]);
    let df: Result<DataFrame<f64, String>> = DataFrame::new(a).columns(&["a", "b"]);
    assert!(df.is_ok())
}

#[test]
fn dataframe_creation_failure() {
    let a = Array::random((2, 5), Range::new(0., 10.));
    let df: Result<DataFrame<f64, String>> = DataFrame::new(a).columns(&["1", "2"]);
    assert!(df.is_err())
}

#[test]
fn dataframe_select() {
    let a = arr2(&[[2., 3.], [3., 4.]]);
    let df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    let select_idx = vec!["a"];
    let z = df.select(&select_idx[..], UtahAxis::Column).as_df();
    let col = df.data.column(0).clone();
    let expected = DataFrame::from_array(col.to_owned(), UtahAxis::Column)
        .columns(&["a"])
        .unwrap();
    assert_eq!(z.unwrap(), expected);

    let select_idx = vec!["0"];
    let z = df.select(&select_idx[..], UtahAxis::Row).as_df();
    let col = df.data.row(0).clone();
    let expected = DataFrame::from_array(col.to_owned(), UtahAxis::Row)
        .index(&["0"])
        .unwrap()
        .columns(&["a", "b"])
        .unwrap();
    assert_eq!(z.unwrap(), expected);
}

#[test]
fn dataframe_remove() {
    let a = arr2(&[[2., 3.], [3., 4.]]);
    let df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    let remove_idx = vec!["a"];
    let z = df.remove(&remove_idx[..], UtahAxis::Column).as_df();
    let col = df.data.column(1).clone();
    let expected = DataFrame::from_array(col.to_owned(), UtahAxis::Column)
        .columns(&["b"])
        .unwrap();
    assert_eq!(z.unwrap(), expected);

    let remove_idx = vec!["0"];
    let z = df.remove(&remove_idx[..], UtahAxis::Row).as_df();
    let row = df.data.row(1).clone();
    let expected = DataFrame::from_array(row.to_owned(), UtahAxis::Row)
        .index(&["1"])
        .unwrap()
        .columns(&["a", "b"])
        .unwrap();
    assert_eq!(z.unwrap(), expected);
}

#[test]
fn dataframe_append() {
    {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let col = df.select(&["a"], UtahAxis::Column).as_array().unwrap();
        let z = df.append("c", col.view(), UtahAxis::Column).as_df();
        let b = arr2(&[[2., 3., 2.], [3., 3., 4.]]);
        let expected = DataFrame::new(b)
            .columns(&["c", "a", "b"])
            .unwrap();
        assert_eq!(z.unwrap(), expected);
    }
    {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let col = df.select(&["1"], UtahAxis::Row).as_array().unwrap();
        let z = df.append("2", col.view(), UtahAxis::Row).as_df();
        let b = arr2(&[[3., 4.], [2., 3.], [3., 4.]]);
        let expected = DataFrame::new(b)
            .columns(&["a", "b"])
            .unwrap()
            .index(&["2", "0", "1"])
            .unwrap();
        assert_eq!(z.unwrap(), expected);
    }
}

#[test]
fn dataframe_mapdf() {
    {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let _ = df.map(|x| x * 2.0, UtahAxis::Row).as_df();
        let b = arr2(&[[4., 6.], [6., 8.]]);
        let expected: DataFrame<f64, String> = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(df, expected);
    }
    {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let _ = df.map(|x| x * 2.0, UtahAxis::Column).as_df();
        let b = arr2(&[[4., 6.], [6., 8.]]);
        let expected: DataFrame<f64, String> = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(df, expected);
    }
}

#[test]
fn dataframe_mean() {
    {
        let a = arr2(&[[2., 6.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<f64, String> = df.mean(UtahAxis::Row).as_df().unwrap();
        let b = arr2(&[[4.], [3.5]]);
        let expected = DataFrame::new(b);
        assert_eq!(z, expected);
    }
    {
        let a = arr2(&[[2., 6.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<f64, String> = df.mean(UtahAxis::Column).as_df().unwrap();
        let b = arr2(&[[2.5, 5.]]);
        let expected = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(z, expected);
    }
}

#[test]
fn dataframe_sum() {
    {
        let a = arr2(&[[2., 6.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<f64, String> = df.sumdf(UtahAxis::Row).as_df().unwrap();
        let b = arr2(&[[8.], [7.]]);
        let expected = DataFrame::new(b);
        assert_eq!(z, expected);
    }
    {
        let a = arr2(&[[2., 6.], [3., 4.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<f64, String> = df.sumdf(UtahAxis::Column).as_df().unwrap();
        let b = arr2(&[[5., 10.]]);
        let expected = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(z, expected);
    }
}


#[test]
fn dataframe_max() {

    {
        let a = arr2(&[[2, 6], [3, 4]]);
        let mut df: DataFrame<i32, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<i32, String> = df.maxdf(UtahAxis::Row).as_df().unwrap();
        let b = arr2(&[[6], [4]]);
        let expected = DataFrame::new(b);
        assert_eq!(z, expected);
    }
    {
        let a = arr2(&[[2, 6], [3, 4]]);
        let mut df: DataFrame<i32, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<i32, String> = df.maxdf(UtahAxis::Column).as_df().unwrap();
        let b = arr2(&[[3, 6]]);
        let expected = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(z, expected);
    }
}

#[test]
fn dataframe_min() {
    {
        let a = arr2(&[[2, 6], [3, 4]]);
        let mut df: DataFrame<i32, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<i32, String> = df.mindf(UtahAxis::Column).as_df().unwrap();
        let b = arr2(&[[2, 4]]);
        let expected = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(z, expected);
    }
    {
        let a = arr2(&[[2, 6], [3, 4]]);
        let mut df: DataFrame<i32, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: DataFrame<i32, String> = df.mindf(UtahAxis::Row).as_df().unwrap();
        let b = arr2(&[[2], [3]]);
        let expected = DataFrame::new(b);
        assert_eq!(z, expected);
    }
}


#[test]
fn dataframe_impute() {
    {
        let a = arr2(&[[2., NAN], [3., 8.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let _ = df.impute(ImputeStrategy::Mean, UtahAxis::Column).as_df();
        let b = arr2(&[[2., 8.], [3., 8.]]);
        let expected: DataFrame<f64, String> = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(df, expected);
    }
    {
        let a = arr2(&[[2., NAN], [3., 8.]]);
        let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let _ = df.impute(ImputeStrategy::Mean, UtahAxis::Row).as_df();
        let b = arr2(&[[2., 2.], [3., 8.]]);
        let expected: DataFrame<f64, String> = DataFrame::new(b).columns(&["a", "b"]).unwrap();
        assert_eq!(df, expected);
    }
}

#[test]
fn dataframe_macro() {
    let k: DataFrame<i32, String> = dataframe!(
      {
      "a" =>  column!([2, 3, 5]),
      "b" =>  column!([2, 0, 6])
      });
    let a = arr2(&[[2, 2], [3, 0], [5, 6]]);
    let df: DataFrame<i32, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    assert_eq!(k, df)
}
// #[test]
// fn read_csv() {
//     {
//         let df: Result<DataFrame<InnerType, OuterType>> = DataFrame::read_csv("./test.csv");
//         let b =
//             arr2(&[[InnerType::Float(8.), InnerType::Str("b".to_string()), InnerType::Float(4.)]]);
//         let expected = DataFrame::new(b).columns(&["a", "b", "c"]).unwrap();
//         assert_eq!(df.unwrap(), expected);
//     }
//
// }
