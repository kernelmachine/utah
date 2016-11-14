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
    use std::collections::HashMap;
    #[test]
    fn outer_left_join() {
        let mut left = HashMap::new();
        left.insert(1, "Alice");
        left.insert(2, "Bob");

        let mut right = HashMap::new();
        right.insert(1, "Programmer");
        let mut res: Vec<(usize, &str, Option<&str>)> =
            Join::new(JoinType::OuterLeftJoin, left.into_iter(), right).collect();
        res.sort_by_key(|x| x.1);
        assert_eq!(res,
                   vec![(1, "Alice", Some("Programmer")), (2, "Bob", None)])

    }
    #[test]
    fn inner_join() {
        let mut left = HashMap::new();
        left.insert(1, "Alice");
        left.insert(2, "Bob");
        left.insert(3, "Suchin");

        let mut right = HashMap::new();
        right.insert(1, "Programmer");
        right.insert(3, "Data Scientist");
        let mut res: Vec<(usize, &str, Option<&str>)> =
            Join::new(JoinType::InnerJoin, left.into_iter(), right).collect();
        res.sort_by_key(|x| x.1);
        assert_eq!(res,
                   vec![(1, "Alice", Some("Programmer")), (3, "Suchin", Some("Data Scientist"))])

    }

    #[test]
    fn dataframe_creation() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::new(a, names);
        assert!(df.is_ok())
    }

    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = {
            DataFrame::new(a, names).unwrap()
        };
        assert!(df.get("a".to_string()) == Ok(arr2(&[[2., 3.], [3., 4.]]).column(0).to_owned()))
    }

    #[test]
    fn dataframe_join() {

        let a = Array::random((3, 2), Range::new(0., 10.));
        let b = arr2(&[[1.], [5.], [2.]]);
        let f = arr2(&[[1.], [0.], [2.]]);
        let d = Array::random((3, 2), Range::new(0., 10.));
        let c = stack(Axis(1), &[a.view(), b.view()]).unwrap();
        let e = stack(Axis(1), &[d.view(), f.view()]).unwrap();

        let c_names: Vec<String> = vec!["1", "2", "3"].iter().map(|x| x.to_string()).collect();
        let e_names: Vec<String> = vec!["4", "5", "3"].iter().map(|x| x.to_string()).collect();


        let c_df = DataFrame::new(c, c_names).unwrap();
        let e_df = DataFrame::new(e, e_names).unwrap();

        println!("c - {:?}", c_df);
        println!("e - {:?}", e_df);

        println!("join - {:?}", c_df.inner_join(e_df, "3"));


    }


    #[test]
    fn dataframe_insert() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::new(a, names).unwrap();
        let new_array = arr2(&[[5.], [6.]]);

        let new_df = df.insert(new_array, "c".to_string());
        let new_names: Vec<String> = vec!["a", "b", "c"].iter().map(|x| x.to_string()).collect();
        let a_prime = arr2(&[[2., 3., 5.], [3., 4., 6.]]);

        assert_eq!(DataFrame::new(a_prime, new_names).unwrap(), new_df.unwrap())
    }

    #[test]
    fn dataframe_creation_failure() {
        let a = Array::random((2, 5), Range::new(0., 10.));
        let names = vec!["1", "2"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::new(a, names);
        assert!(df.is_err())
    }



    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        let names: Vec<String> =
            vec!["1", "2", "3", "4", "5"].iter().map(|x| x.to_string()).collect();
        b.iter(|| DataFrame::new(a.clone(), names.clone()));
    }



    #[bench]
    fn bench_inner_join(b: &mut Bencher) {
        let a = Array::random((2000, 10), Range::new(0., 10.));
        let z = Array::random((2000, 1), Range::new(0., 10.));
        let d = Array::random((2000, 10), Range::new(0., 10.));
        let c = stack(Axis(1), &[a.view(), z.view()]).unwrap();
        let e = stack(Axis(1), &[d.view(), z.view()]).unwrap();


        let c_names: Vec<String> = (0..11).map(|x| x.to_string()).collect();
        let e_names: Vec<String> = (0..11).map(|x| x.to_string()).collect();


        let c_df = DataFrame::new(c, c_names).unwrap();
        let e_df = DataFrame::new(e, e_names).unwrap();

        b.iter(|| c_df.clone().inner_join(e_df.clone(), "9"));
    }

    #[bench]
    fn bench_stack(b: &mut Bencher) {
        let a = Array::random((2000, 10), Range::new(0., 10.));
        let d = Array::random((2000, 10), Range::new(0., 10.));


        b.iter(|| stack(Axis(1), &[a.view(), d.view()]));
    }

    #[bench]
    fn bench_inner_join_bare(b: &mut Bencher) {
        let mut left = HashMap::new();
        for (i, j) in (1..2000).zip((1..2000).map(|x| x.to_string())) {
            left.insert(i, j);
        }

        let mut right = HashMap::new();
        for (i, j) in (1993..4000).zip((1993..4000).map(|x| x.to_string())) {
            right.insert(i, j);
        }

        b.iter(|| {
            let res: Vec<(i32, String, Option<String>)> =
                Join::new(JoinType::InnerJoin, left.clone().into_iter(), right.clone()).collect();
            res
        });

    }

}
