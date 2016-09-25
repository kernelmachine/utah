mod tests {

    extern crate rand;
    extern crate test;
    use ndarray::{arr2, arr1};
    use dataframe::*;
    use test::Bencher;
    use ndarray::Array;
    use rand::distributions::Range;
    use ndarray_rand::RandomExt;
    use std::rc::Rc;

    #[test]
    fn dataframe_creation() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"];
        let df = DataFrame::from_array(&a, &names);
        assert!(df.is_ok())
    }

    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"];
        let df = DataFrame::from_array(&a, &names).unwrap();
        assert!(df.get("a") == Some(&a.column(0).to_owned()))
    }

    #[test]
    fn dataframe_join() {
        let a = arr2(&[[2., 3.], [3., 4.], [7., 34.]]);
        let names = vec!["a", "b"];
        let names1 = vec!["a", "c"];

        let z = arr2(&[[2., 3.], [7., 8.]]);
        let df = DataFrame::from_array(&a, &names).unwrap();
        let df1 = DataFrame::from_array(&z, &names1).unwrap();
        println!("{:?}", df.inner_join(&df1, "a"));
        assert!(df.inner_join(&df1, "a").is_ok());
    }

    #[test]
    fn dataframe_join_fails() {
        let a = arr2(&[[1., 3.], [3., 4.], [8., 34.]]);
        let names = vec!["a", "b"];
        let names1 = vec!["a", "c"];

        let z = arr2(&[[2., 3.], [7., 8.]]);
        let df = DataFrame::from_array(&a, &names).unwrap();
        let df1 = DataFrame::from_array(&z, &names1).unwrap();
        assert!(df.inner_join(&df1, "a").is_err());
    }
    // #[test]
    // fn dataframe_insert() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //     let names = vec!["a", "b"];
    //     let mut df = from_array(&a, &names).unwrap();
    //     let new_array = Rc::new(arr1(&[5., 6.]));
    //     df.insert("c", new_array.clone());
    //     let new_names = vec!["a", "b", "c"];
    //     let a_prime = arr2(&[[2., 3.], [3., 4.], [5., 6.]]);
    //     assert!(from_array(&a, &names).unwrap() == df)
    // }

    #[test]
    fn it_fails() {
        let a = Array::random((2, 5), Range::new(0., 10.));
        let names = vec!["1", "2"];
        let df = DataFrame::from_array(&a, &names);
        assert!(df.is_err())
    }




    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10000, 5), Range::new(0., 10.));
        let names = vec!["1", "2", "3", "4", "5"];
        b.iter(|| DataFrame::from_array(&a, &names));
    }
    #[bench]
    fn bench_get(b: &mut Bencher) {
        let a = Array::random((100000, 5), Range::new(0., 10.));
        let names = vec!["1", "2", "3", "4", "5"];
        let df = DataFrame::from_array(&a, &names).unwrap();
        b.iter(|| df.get("1"));
    }
    #[bench]
    fn bench_join(b: &mut Bencher) {
        let a = arr2(&[[2., 3.], [3., 4.], [7., 34.]]);
        let names = vec!["a", "b"];
        let names1 = vec!["a", "c"];

        let z = arr2(&[[2., 3.], [7., 8.]]);
        let df = DataFrame::from_array(&a, &names).unwrap();
        let df1 = DataFrame::from_array(&z, &names1).unwrap();
        b.iter(|| df.inner_join(&df1, "a"));
    }
}
