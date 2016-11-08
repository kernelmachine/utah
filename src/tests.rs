mod tests {

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
    fn dataframe_creation() {
        let a = arr2(&[[Data::Float(2.), Data::Float(3.)], [Data::Float(3.), Data::Float(4.)]]);
        let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
        let df = DataFrame::new(a, names);
        assert!(df.is_ok())
    }
    // #[test]
    // fn dataframe_index() {
    //     let a = arr2(&[[2., 3.], [3., 4.]]);
    //     let names = vec!["a", "b"].iter().map(|x| x.to_string()).collect();
    //     let df = DataFrame::from_array(&a, &names).unwrap();
    //     assert!(df.get("a".to_string()) == Some(&a.column(0).to_owned()))
    // }
    //
    #[test]
    fn dataframe_join() {
        let mut rng = thread_rng();

        let a = Array::random((2, 2), Range::new(0., 10.));
        let b = Array::random((2, 1), Range::new(0., 10.));
        let d = Array::random((2, 2), Range::new(0., 10.));
        let c = stack(Axis(1), &[a.view(), b.view()]).unwrap();
        let e = stack(Axis(1), &[d.view(), b.view()]).unwrap();

        // let mut a_names: Vec<String> = (0..2).map(|x| x.to_string()).collect();
        // let mut b_names: Vec<String> = (1..3).map(|x| x.to_string()).collect();
        let mut c_names: Vec<String> = vec!["1", "2", "3"].iter().map(|x| x.to_string()).collect();
        let mut e_names: Vec<String> = vec!["4", "5", "3"].iter().map(|x| x.to_string()).collect();


        let c_df = DataFrame::new(c.mapv(Data::Float), c_names).unwrap();
        let e_df = DataFrame::new(e.mapv(Data::Float), e_names).unwrap();
        // let c_df = DataFrame::new(c_prime.mapv(Data::Float), c_names).unwrap();

        println!("c - {:?}", c_df);
        println!("e - {:?}", e_df);
        // println!("c - {:?}", c_df);
        println!("join - {:?}", c_df.inner_join(e_df, "3"));

        // let mut names: Vec<String> = (0..1).map(|x| x.to_string()).collect();
        // let slice = names.as_mut_slice();
        // rng.shuffle(slice);
        // let names = slice.to_vec();
        //
        // let mut names1: Vec<String> = (0..1).map(|x| x.to_string()).collect();
        // let slice = names1.as_mut_slice();
        // rng.shuffle(slice);
        // let names1 = slice.to_vec();
        // let z = Array::random((2, 2), Range::new(0., 10.));
        //
        // a.column
        // let df = DataFrame::new(a.mapv(Data::Float), names).unwrap();
        // let df1 = DataFrame::new(z.mapv(Data::Float), names1).unwrap();
        // println!("{:?}", df);
        // println!("{:?}", df1);


        // println!("{:?}", df.inner_join(df1, "0"));

        // assert!(df.inner_join(df1, "0").is_ok());
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

    // #[test]
    // fn it_fails() {
    //     let a = Array::random((2, 5), Range::new(0., 10.));
    //     let names = vec!["1", "2"].iter().map(|x| x.to_string()).collect();
    //     let df = DataFrame::from_array(&a, &names);
    //     assert!(df.is_err())
    // }
    //


    #[bench]
    fn bench_creation(b: &mut Bencher) {
        let a = Array::random((10, 5), Range::new(0., 10.));
        let names: Vec<String> =
            vec!["1", "2", "3", "4", "5"].iter().map(|x| x.to_string()).collect();
        b.iter(|| DataFrame::new(a.mapv(Data::Float), names.clone()));
    }
    // #[bench]
    // fn bench_get(b: &mut Bencher) {
    //     let a = Array::random((50, 5), Range::new(0., 10.));
    //     let names = vec!["1", "2", "3", "4", "5"].iter().map(|x| x.to_string()).collect();
    //     let df = DataFrame::from_array(&a, &names).unwrap();
    //     b.iter(|| df.get("1".to_string()));
    // }
    #[bench]
    fn bench_join(b: &mut Bencher) {
        let mut rng = thread_rng();

        let a = Array::random((200, 10), Range::new(0., 10.));
        let z = Array::random((200, 1), Range::new(0., 10.));
        let d = Array::random((200, 10), Range::new(0., 10.));
        let c = stack(Axis(1), &[a.view(), z.view()]).unwrap();
        let e = stack(Axis(1), &[d.view(), z.view()]).unwrap();

        // let mut a_names: Vec<String> = (0..2).map(|x| x.to_string()).collect();
        // let mut b_names: Vec<String> = (1..3).map(|x| x.to_string()).collect();
        let mut c_names: Vec<String> = (0..11).map(|x| x.to_string()).collect();
        let mut e_names: Vec<String> = (0..11).map(|x| x.to_string()).collect();


        let c_df = DataFrame::new(c.mapv(Data::Float), c_names).unwrap();
        let e_df = DataFrame::new(e.mapv(Data::Float), e_names).unwrap();
        // let c_df = DataFrame::new(c_prime.mapv(Data::Float), c_names).unwrap();
        //
        // println!("c - {:?}", c_df);
        // println!("e - {:?}", e_df);
        // // println!("c - {:?}", c_df);
        // println!("join - {:?}", c_df.inner_join(e_df, "3"));
        b.iter(|| c_df.clone().inner_join(e_df.clone(), "9"));
    }
}
