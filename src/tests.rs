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
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("a".to_string(), 0);
        names.insert("b".to_string(), 1);
        let df = DataFrame::new(a, names);
        assert!(df.is_ok())
    }

    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("a".to_string(), 0);
        names.insert("b".to_string(), 1);
        let df = DataFrame::new(a, names).unwrap();
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

        let mut c_names: BTreeMap<String, usize> = BTreeMap::new();
        c_names.insert("1".to_string(), 0);
        c_names.insert("2".to_string(), 1);
        c_names.insert("3".to_string(), 2);

        let mut e_names: BTreeMap<String, usize> = BTreeMap::new();
        e_names.insert("4".to_string(), 0);
        e_names.insert("5".to_string(), 1);
        e_names.insert("3".to_string(), 2);



        let c_df = DataFrame::new(c.clone(), c_names).unwrap();
        let e_df = DataFrame::new(e.clone(), e_names).unwrap();



        let mut join_names: BTreeMap<String, usize> = BTreeMap::new();
        join_names.insert("1".to_string(), 0);
        join_names.insert("2".to_string(), 1);
        join_names.insert("3".to_string(), 2);
        join_names.insert("4_x".to_string(), 3);
        join_names.insert("5_x".to_string(), 4);
        join_names.insert("3_x".to_string(), 5);
        let join_matrix = stack(Axis(1),
                                &[c.select(Axis(0), &[0, 2]).view(),
                                  e.select(Axis(0), &[0, 2]).view()])
            .unwrap();
        let join_df = DataFrame::new(join_matrix, join_names);
        let test_df = c_df.inner_join(&e_df, "3");
        println!("{:?}", test_df.clone().unwrap().data_map);
        assert_eq!(join_df.unwrap(), test_df.clone().unwrap())

    }


    #[test]
    fn dataframe_insert() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("a".to_string(), 0);
        names.insert("b".to_string(), 1);
        let df = DataFrame::new(a, names).unwrap();
        let new_array = arr2(&[[5.], [6.]]);

        let new_df = df.insert(new_array, "c".to_string());
        let mut new_names: BTreeMap<String, usize> = BTreeMap::new();
        new_names.insert("a".to_string(), 0);
        new_names.insert("b".to_string(), 1);
        new_names.insert("c".to_string(), 2);
        let a_prime = arr2(&[[2., 3., 5.], [3., 4., 6.]]);

        assert_eq!(DataFrame::new(a_prime, new_names).unwrap(), new_df.unwrap())
    }

    #[test]
    fn dataframe_concat() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let b = arr2(&[[7., 1.], [7., 6.]]);
        let mut a_names: BTreeMap<String, usize> = BTreeMap::new();
        a_names.insert("a".to_string(), 0);
        a_names.insert("b".to_string(), 1);

        let mut b_names: BTreeMap<String, usize> = BTreeMap::new();
        b_names.insert("c".to_string(), 0);
        b_names.insert("d".to_string(), 1);
        let df = DataFrame::new(a, a_names).unwrap();
        let df_1 = DataFrame::new(b, b_names).unwrap();

        let row_concat = df.concat(Axis(0), &df_1);
        let col_concat = df.concat(Axis(1), &df_1);


        let mut a_prime_names: BTreeMap<String, usize> = BTreeMap::new();
        a_prime_names.insert("a".to_string(), 0);
        a_prime_names.insert("b".to_string(), 1);
        let mut b_prime_names: BTreeMap<String, usize> = BTreeMap::new();
        b_prime_names.insert("a".to_string(), 0);
        b_prime_names.insert("b".to_string(), 1);
        b_prime_names.insert("c".to_string(), 2);
        b_prime_names.insert("d".to_string(), 3);
        let a_prime = arr2(&[[2., 3.], [3., 4.], [7., 1.], [7., 6.]]);
        let b_prime = arr2(&[[2., 3., 7., 1.], [3., 4., 7., 6.]]);

        assert_eq!(DataFrame::new(a_prime, a_prime_names).unwrap(),
                   row_concat.unwrap());
        assert_eq!(DataFrame::new(b_prime, b_prime_names).unwrap(),
                   col_concat.unwrap())
    }

    #[test]
    fn dataframe_drop() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("a".to_string(), 0);
        names.insert("b".to_string(), 1);
        let mut df = DataFrame::new(a, names).unwrap();
        let new_df = df.drop(&["a".to_string()]);
        let mut new_names: BTreeMap<String, usize> = BTreeMap::new();
        new_names.insert("b".to_string(), 0);
        let a_prime = arr2(&[[3.], [4.]]);
        assert_eq!(DataFrame::new(a_prime, new_names).unwrap(), new_df.unwrap())
    }

    #[test]
    fn dataframe_creation_failure() {
        let a = Array::random((2, 5), Range::new(0., 10.));
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("1".to_string(), 0);
        names.insert("2".to_string(), 1);
        let df = DataFrame::new(a, names);
        assert!(df.is_err())
    }



    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        names.insert("1".to_string(), 0);
        names.insert("2".to_string(), 1);
        names.insert("3".to_string(), 2);
        names.insert("4".to_string(), 3);
        names.insert("5".to_string(), 4);
        b.iter(|| DataFrame::new(a.clone(), names.clone()));
    }



    #[bench]
    fn bench_inner_join(b: &mut Bencher) {
        let a = Array::random((20000, 10), Range::new(0., 10.));
        let z = Array::random((20000, 1), Range::new(0., 10.));
        let d = Array::random((20000, 10), Range::new(0., 10.));
        let c = stack(Axis(1), &[a.view(), z.view()]).unwrap();
        let e = stack(Axis(1), &[d.view(), z.view()]).unwrap();


        let mut c_names: BTreeMap<String, usize> = BTreeMap::new();

        for i in 0..11 {
            c_names.insert(i.to_string(), i);
        }
        let mut e_names: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..11 {
            e_names.insert(i.to_string(), i);
        }


        let c_df = DataFrame::new(c, c_names).unwrap();
        let e_df = DataFrame::new(e, e_names).unwrap();

        b.iter(|| c_df.inner_join(&e_df, "9"));
    }

    #[bench]
    fn bench_stack(b: &mut Bencher) {
        let a = Array::random((20000, 10), Range::new(0., 10.));
        let d = Array::random((20000, 10), Range::new(0., 10.));


        b.iter(|| stack(Axis(1), &[a.view(), d.view()]));
    }

    #[bench]
    fn bench_inner_join_bare(b: &mut Bencher) {
        let mut left = HashMap::new();
        for (i, j) in (1..20000).zip((1..20000).map(|x| x.to_string())) {
            left.insert(i, j);
        }

        let mut right = HashMap::new();
        for (i, j) in (19993..40000).zip((19993..40000).map(|x| x.to_string())) {
            right.insert(i, j);
        }

        b.iter(|| {
            let res: Vec<(i32, String, Option<String>)> =
                Join::new(JoinType::InnerJoin, left.clone().into_iter(), right.clone()).collect();
            res
        });

    }

}
