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
    use rand::{thread_rng, Rng};

    #[test]
    fn dataframe_creation() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::from_array(&a, &names);
        assert!(df.is_ok())
    }

    #[test]
    fn dataframe_index() {
        let a = arr2(&[[2., 3.], [3., 4.]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::from_array(&a, &names).unwrap();
        assert!(df.get("a".to_string()) == Some(&a.column(0).to_owned()))
    }

    #[test]
    fn dataframe_join() {
        let mut rng = thread_rng();

        let a = Array::random((100000, 50), Range::new(0., 10.));
        let mut names: Vec<String> = (0..50).map(|x| x.to_string()).collect();
        let slice = names.as_mut_slice();
        rng.shuffle(slice);
        let names = slice.to_vec();

        let mut names1: Vec<String> = (40..90).map(|x| x.to_string()).collect();
        let slice = names1.as_mut_slice();
        rng.shuffle(slice);
        let names1 = slice.to_vec();
        let z = Array::random((100000, 50), Range::new(0., 10.));
        let df = DataFrame::from_array(&a, &names).unwrap();
        let df1 = DataFrame::from_array(&z, &names1).unwrap();
        assert!(df.inner_join(&df1, "40").is_ok())
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
        let names = vec!["1", "2"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::from_array(&a, &names);
        assert!(df.is_err())
    }




    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        let names = vec!["1", "2", "3", "4", "5"].iter().map(|x| x.to_string()).collect();
        b.iter(|| DataFrame::from_array(&a, &names));
    }
    #[bench]
    fn bench_get(b: &mut Bencher) {
        let a = Array::random((50, 5), Range::new(0., 10.));
        let names = vec!["1", "2", "3", "4", "5"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::from_array(&a, &names).unwrap();
        b.iter(|| df.get("1".to_string()));
    }
    #[bench]
    fn bench_join(b: &mut Bencher) {
        let mut rng = thread_rng();

        let a = Array::random((100000, 50), Range::new(0., 10.));
        let mut names: Vec<String> = (0..50).map(|x| x.to_string()).collect();
        let slice = names.as_mut_slice();
        rng.shuffle(slice);
        let names = slice.to_vec();

        let mut names1: Vec<String> = (40..90).map(|x| x.to_string()).collect();
        let slice = names1.as_mut_slice();
        rng.shuffle(slice);
        let names1 = slice.to_vec();
        let z = Array::random((100000, 50), Range::new(0., 10.));
        let df = DataFrame::from_array(&a, &names).unwrap();
        let df1 = DataFrame::from_array(&z, &names1).unwrap();
        b.iter(|| df.inner_join(&df1, "40").is_ok())
    }
}
