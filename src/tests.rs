#[allow(unused_imports)]

pub mod tests {
    extern crate rand;
    extern crate test;
    use ndarray::{arr2, arr1, Axis, stack};
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
    #[test]
    fn outer_left_join() {
        let a = arr2(&[["Alice"], ["Bob"]]);
        let left = DataFrame::new(a).index(&[1, 2]).unwrap();
        let b = arr2(&[["Programmer"]]);
        let right = DataFrame::new(b).index(&[1]).unwrap();
        let first_index = &left.index[0];
        let second_index = &left.index[1];
        println!("{:?}", second_index);
        let res: Vec<_> = left.outer_left_join(&right, Axis(0)).collect();
        assert_eq!(res,
                   vec![(first_index.to_owned(),
                         left.data.row(0).view(),
                         Some(right.data.row(0).view())),
                        (second_index.to_owned(), left.data.row(1).view(), None)])

    }
    #[test]
    fn inner_join() {
        let a = arr2(&[["Alice"], ["Bob"], ["Suchin"]]);
        let left = DataFrame::new(a).index(&[1, 2, 3]).unwrap();
        let b = arr2(&[["Programmer"], ["Data Scientist"]]);
        let right = DataFrame::new(b).index(&[1, 3]).unwrap();
        let first_index = &left.index[0];
        let second_index = &left.index[2];
        println!("{:?}", second_index);
        let res: Vec<_> = left.inner_left_join(&right, Axis(0)).collect();
        assert_eq!(res,
                   vec![(first_index.to_owned(),
                         left.data.row(0).view(),
                         right.data.row(0).view()),
                        (second_index.to_owned(),
                         left.data.row(2).view(),
                         right.data.row(1).view())])
    }
    #[test]
    fn dataframe_creation() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let df = DataFrame::new(a).columns(&["a", "b"]);
        assert!(df.is_ok())
    }
    //
    #[test]
    fn dataframe_creation_datetime_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);

        let df: Result<DataFrame> = DataFrame::new(a)
            .columns(&[UTC.ymd(2014, 7, 8).and_hms(9, 10, 11),
                       UTC.ymd(2014, 10, 5).and_hms(2, 5, 7)]);
        assert!(df.is_ok())
    }
    #[test]
    fn dataframe_creation_mixed_types() {
        let a = arr2(&[[InnerType::Str("string".to_string()), InnerType::Int64(1)],
                       [InnerType::Float(4.), InnerType::Int32(4)]]);

        let df: Result<DataFrame> = DataFrame::new(a)
            .columns(&[UTC.ymd(2014, 7, 8).and_hms(9, 10, 11),
                       UTC.ymd(2014, 10, 5).and_hms(2, 5, 7)]);
        assert!(df.is_ok())
    }
    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);

        let df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
        let z: Vec<_> = df.select(vec![OuterType::Str("a".to_string())], Axis(1)).collect();
        assert!(z[0].1 == arr2(&[[2., 3.], [3., 4.]]).mapv(InnerType::from).column(0).to_owned())
    }



    #[test]
    fn dataframe_creation_failure() {
        let a = Array::random((2, 5), Range::new(0., 10.));
        let df = DataFrame::new(a).columns(&["1", "2"]);
        assert!(df.is_err())
    }



    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        b.iter(|| DataFrame::new(a.clone()).columns(&["1", "2", "3", "4", "5"]));
    }



    #[bench]
    fn test_inner_join(b: &mut Bencher) {
        let c = Array::random((20000, 10), Range::new(0., 10.));
        let e = Array::random((20000, 10), Range::new(0., 10.));


        let mut c_names: Vec<String> = vec![];
        for i in 0..10 {
            c_names.push(i.to_string());
        }

        let mut e_names: Vec<String> = vec![];
        for i in 0..10 {
            e_names.push(i.to_string());
        }

        let mut c_index: Vec<String> = vec![];
        for i in 0..20000 {
            c_index.push(i.to_string());
        }

        let mut e_index: Vec<String> = vec![];
        for i in 1999..21999 {
            e_index.push(i.to_string());
        }

        let c_df = DataFrame::new(c)
            .columns(&c_names[..])
            .unwrap()
            .index(&c_index[..])
            .unwrap();
        let e_df = DataFrame::new(e)
            .columns(&e_names[..])
            .unwrap()
            .index(&e_index[..])
            .unwrap();
        b.iter(|| {
            let _: Vec<_> = c_df.inner_left_join(&e_df, Axis(0)).collect();
        });
    }


}
