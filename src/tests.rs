
#![feature(test)]
#[cfg(test)]
mod tests {

    extern crate rand;
    extern crate test;
    use ndarray::{arr2, arr1};
    use super::*;
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
        assert!(df.get("a") == Some(&a.column(0)))
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
}
