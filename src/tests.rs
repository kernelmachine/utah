#[allow(unused_imports)]

pub mod tests {
    extern crate rand;
    extern crate test;
    use ndarray::{arr2, arr1, stack};
    use dataframe::*;
    use test::Bencher;
    use ndarray::Array;
    use rand::distributions::Range;
    use ndarray_rand::RandomExt;
    use std::rc::Rc;
    use rand::{thread_rng, Rng};
    use std::collections::{HashMap, BTreeMap};
    use chrono::*;
    use types::*;
    use from::*;
    use error::*;
    use aggregate::*;
    use transform::*;
    use traits::*;
    use std::f64::NAN;
    #[test]
    fn outer_left_join() {
        let a = arr2(&[["Alice"], ["Bob"]]);
        let left: DataFrame<InnerType, OuterType> =
            DataFrame::new(a).index(&[1, 2]).unwrap().columns(&["a"]).unwrap();
        let b = arr2(&[["Programmer"]]);
        let right: DataFrame<InnerType, OuterType> =
            DataFrame::new(b).index(&[1]).unwrap().columns(&["b"]).unwrap();
        let first_index = &left.index[0];
        let second_index = &left.index[1];
        let res = left.outer_left_join(&right).as_df();
        let expected_data = arr2(&[[InnerType::Str(String::from("Alice")),
                                    InnerType::Str(String::from("Programmer"))],
                                   [InnerType::Str(String::from("Bob")), InnerType::Empty]]);
        let expected =
            DataFrame::new(expected_data).index(&[1, 2]).unwrap().columns(&["a", "b"]).unwrap();
        assert_eq!(res, expected);

    }
    #[test]
    fn inner_join() {
        let a = arr2(&[["Alice"], ["Bob"], ["Suchin"]]);
        let left: DataFrame<InnerType, OuterType> =
            DataFrame::new(a).index(&[1, 2, 3]).unwrap().columns(&["a"]).unwrap();
        let b = arr2(&[["Programmer"], ["Data Scientist"]]);
        let right = DataFrame::new(b).index(&[1, 3]).unwrap().columns(&["b"]).unwrap();
        let first_index = &left.index[0];
        let second_index = &left.index[2];
        let res = left.inner_left_join(&right).as_df();
        let expected_data = arr2(&[[InnerType::Str(String::from("Alice")),
                                    InnerType::Str(String::from("Programmer"))],
                                   [InnerType::Str(String::from("Suchin")),
                                    InnerType::Str(String::from("Data Scientist"))]]);
        let expected =
            DataFrame::new(expected_data).index(&[1, 3]).unwrap().columns(&["a", "b"]).unwrap();
        assert_eq!(res, expected);
    }

    #[test]
    fn dataframe_creation() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let df: Result<DataFrame<f64, String>> = DataFrame::new(a).columns(&["a", "b"]);
        assert!(df.is_ok())
    }

    // #[test]
    // fn dataframe_creation_datetime_index() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //
    //     let df: Result<DataFrame<f64, DateTime<UTC>>> = DataFrame::new(a)
    //         .columns(&[UTC.ymd(2014, 7, 8).and_hms(9, 10, 11),
    //                    UTC.ymd(2014, 10, 5).and_hms(2, 5, 7)]);
    //     assert!(df.is_ok())
    // }
    #[test]
    fn dataframe_creation_mixed_types() {
        let a = arr2(&[[InnerType::Str("string".to_string()), InnerType::Float(14.)],
                       [InnerType::Float(4.), InnerType::Int32(4)]]);

        let df: Result<DataFrame<InnerType, OuterType>> = DataFrame::new(a)
            .columns(&[UTC.ymd(2014, 7, 8).and_hms(9, 10, 11),
                       UTC.ymd(2014, 10, 5).and_hms(2, 5, 7)]);
        assert!(df.is_ok())
    }
    //
    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let select_idx = vec!["a"];
        let z = df.select(&select_idx[..], UtahAxis::Column).as_df();
        let col = df.data.column(0).clone();
        let expected = DataFrame::from_array(col.to_owned(), UtahAxis::Column)
            .columns(&["a"])
            .unwrap();
        assert_eq!(z, expected);

        let select_idx = vec!["0"];
        let z = df.select(&select_idx[..], UtahAxis::Row).as_df();
        let col = df.data.row(0).clone();
        let expected = DataFrame::from_array(col.to_owned(), UtahAxis::Row)
            .index(&["0"])
            .unwrap()
            .columns(&["a", "b"])
            .unwrap();
        assert_eq!(z, expected);
    }
    #[test]
    fn dataframe_creation_failure() {
        let a = Array::random((2, 5), Range::new(0., 10.));
        let df: Result<DataFrame<f64, String>> = DataFrame::new(a).columns(&["1", "2"]);
        assert!(df.is_err())
    }
    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        b.iter(|| {
            let _: Result<DataFrame<f64, String>> = DataFrame::new(a.clone())
                .columns(&["1", "2", "3", "4", "5"]);
        });
    }



    #[bench]
    fn test_inner_join(b: &mut Bencher) {
        let c = Array::random((200, 10), Range::new(0., 10.));
        let e = Array::random((200, 10), Range::new(0., 10.));


        let mut c_names: Vec<String> = vec![];
        for i in 0..10 {
            c_names.push(i.to_string());
        }

        let mut e_names: Vec<String> = vec![];
        for i in 0..10 {
            e_names.push(i.to_string());
        }

        let mut c_index: Vec<String> = vec![];
        for i in 0..200 {
            c_index.push(i.to_string());
        }

        let mut e_index: Vec<String> = vec![];
        for i in 1..201 {
            e_index.push(i.to_string());
        }

        let mut c_df: DataFrame<f64, String> = DataFrame::new(c)
            .columns(&c_names[..])
            .unwrap()
            .index(&c_index[..])
            .unwrap();
        let e_df: DataFrame<f64, String> = DataFrame::new(e)
            .columns(&e_names[..])
            .unwrap()
            .index(&e_index[..])
            .unwrap();
        b.iter(|| {
            let _: DataFrame<f64, String> = c_df.sumdf(UtahAxis::Column).as_df();
        });
    }


}
