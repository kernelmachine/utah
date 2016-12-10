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
    // #[test]
    // fn outer_left_join() {
    //     let mut left = HashMap::new();
    //     left.insert(1, "Alice");
    //     left.insert(2, "Bob");
    //
    //     let mut right = HashMap::new();
    //     right.insert(1, "Programmer");
    //     let mut res: Vec<(usize, &str, Option<&str>)> =
    //         Join::new(JoinType::OuterLeftJoin, left.into_iter(), right).collect();
    //     res.sort_by_key(|x| x.1);
    //     assert_eq!(res,
    //                vec![(1, "Alice", Some("Programmer")), (2, "Bob", None)])
    //
    // }
    //
    // #[test]
    // fn inner_join() {
    //     let mut left = HashMap::new();
    //     left.insert(1, "Alice");
    //     left.insert(2, "Bob");
    //     left.insert(3, "Suchin");
    //
    //     let mut right = HashMap::new();
    //     right.insert(1, "Programmer");
    //     right.insert(3, "Data Scientist");
    //     let mut res: Vec<(usize, &str, Option<&str>)> =
    //         Join::new(JoinType::InnerJoin, left.into_iter(), right).collect();
    //     res.sort_by_key(|x| x.1);
    //     assert_eq!(res,
    //                vec![(1, "Alice", Some("Programmer")), (3, "Suchin", Some("Data Scientist"))])
    //
    // }
    //
    // #[test]
    // fn dataframe_creation() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //     let df = DataFrame::new(a).columns(&["a", "b"]);
    //     assert!(df.is_ok())
    // }
    //
    // #[test]
    // fn dataframe_creation_datetime_index() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //
    //     let df: Result<DataFrame> = DataFrame::new(a)
    //         .columns(&[UTC.ymd(2014, 7, 8).and_hms(9, 10, 11),
    //                    UTC.ymd(2014, 10, 5).and_hms(2, 5, 7)]);
    //     assert!(df.is_ok())
    // }
    //
    // #[test]
    // fn dataframe_index() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //
    //     let df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    //     assert!(df.get_column("a").unwrap() ==
    //             arr2(&[[2., 3.], [3., 4.]]).mapv(InnerType::from).column(0).to_owned())
    // }
    //
    // #[test]
    // fn dataframe_join() {
    //
    //     let a = Array::random((3, 2), Range::new(0., 10.));
    //     let b = arr2(&[[1.], [5.], [2.]]);
    //     let f = arr2(&[[1.], [0.], [2.]]);
    //     let d = Array::random((3, 2), Range::new(0., 10.));
    //     let c = stack(Axis(1), &[a.view(), b.view()]).unwrap();
    //     let e = stack(Axis(1), &[d.view(), f.view()]).unwrap();
    //
    //     let c_df = DataFrame::new(c.clone())
    //         .columns(&["1", "2", "3"])
    //         .unwrap()
    //         .index(&["1", "2", "3"])
    //         .unwrap();
    //
    //     let e_df = DataFrame::new(e.clone())
    //         .columns(&["4", "5", "6"])
    //         .unwrap()
    //         .index(&["1", "2", "4"])
    //         .unwrap();
    //
    //
    //
    //     let join_matrix = stack(Axis(1),
    //                             &[c.select(Axis(0), &[0, 1]).view(),
    //                               e.select(Axis(0), &[0, 1]).view()])
    //         .unwrap();
    //
    //     let join_df = DataFrame::new(join_matrix)
    //         .columns(&["1", "2", "3", "4_x", "5_x", "6_x"])
    //         .unwrap()
    //         .index(&["1", "2"]);
    //     let test_df = c_df.inner_join(&e_df);
    //     assert_eq!(join_df.unwrap(), test_df.unwrap().clone())
    //
    // }
    //
    //
    //
    // #[test]
    // fn dataframe_insert() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //
    //     let df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    //
    //     let new_array = arr2(&[[5.], [6.]]);
    //
    //     let new_df = df.insert_column(new_array, "c");
    //
    //     let a_prime = arr2(&[[2., 3., 5.], [3., 4., 6.]]);
    //     assert_eq!(DataFrame::new(a_prime).columns(&["a", "b", "c"]).unwrap(),
    //                new_df.unwrap())
    // }
    //
    // #[test]
    // fn dataframe_concat() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //     let b = arr2(&[[7., 1.], [7., 6.]]);
    //
    //
    //     let df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    //     let df_1 = DataFrame::new(b).columns(&["c", "d"]).unwrap();
    //
    //     let row_concat = df.concat(Axis(0), &df_1);
    //     let col_concat = df.concat(Axis(1), &df_1);
    //
    //     let a_prime = arr2(&[[2., 3.], [3., 4.], [7., 1.], [7., 6.]]);
    //     let b_prime = arr2(&[[2., 3., 7., 1.], [3., 4., 7., 6.]]);
    //
    //     assert_eq!(DataFrame::new(a_prime)
    //                    .columns(&["a", "b"])
    //                    .unwrap()
    //                    .index(&["0", "1", "0_x", "1_x"])
    //                    .unwrap(),
    //                row_concat.unwrap());
    //     assert_eq!(DataFrame::new(b_prime).columns(&["a", "b", "c", "d"]).unwrap(),
    //                col_concat.unwrap())
    // }
    //
    // #[test]
    // fn dataframe_drop_column() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //
    //     let mut df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    //     let new_df = df.drop_column(&["a"]);
    //     let a_prime = arr2(&[[3.], [4.]]);
    //     assert_eq!(DataFrame::new(a_prime).columns(&["b"]).unwrap(),
    //                new_df.unwrap())
    // }
    //
    // #[test]
    // fn dataframe_drop_row() {
    //     let a = arr2(&[[2., 3.], [5., 4.]]);
    //
    //     let mut df = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    //     let new_df = df.drop_row(&["1"]);
    //
    //     let a_prime = arr2(&[[2., 3.]]);
    //     assert_eq!(DataFrame::new(a_prime).columns(&["a", "b"]).unwrap(),
    //                new_df.unwrap())
    // }


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
    fn bench_inner_join(b: &mut Bencher) {
        let c = Array::random((200000, 10), Range::new(0., 10.));
        let e = Array::random((200000, 10), Range::new(0., 10.));


        let mut c_names: Vec<String> = vec![];
        for i in 0..10 {
            c_names.push(i.to_string());
        }

        let mut e_names: Vec<String> = vec![];
        for i in 0..10 {
            e_names.push(i.to_string());
        }

        let mut c_index: Vec<String> = vec![];
        for i in 0..200000 {
            c_index.push(i.to_string());
        }

        let mut e_index: Vec<String> = vec![];
        for i in 1999..201999 {
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
            let z: Vec<_> = join(DataFrameIterator::new(&c_df, Axis(0)),
                                 &DataFrameIterator::new(&e_df, Axis(0)))
                .collect();
        });
    }


    #[bench]
    fn bench_inner_join_bare(b: &mut Bencher) {
        let mut left = HashMap::new();
        for (i, j) in (1..20000).zip((1..20000)) {
            left.insert(i, j);
        }

        let mut right = HashMap::new();
        for (i, j) in (19993..40000).zip((19993..40000)) {
            right.insert(i, j);
        }

        b.iter(|| {
            let res: Vec<(i32, i32, Option<i32>)> =
                Join::new(JoinType::InnerJoin, left.clone().into_iter(), right.clone()).collect();
            res
        });

    }

}
