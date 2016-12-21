use traits::ToDataFrame;
use types::*;
use std::iter::Iterator;
use std::iter::repeat;
use std::collections::HashMap;
use ndarray::Array;
use std::hash::Hash;
use std::fmt::Debug;
use dataframe::*;
use std::ops::{Add, Div, Mul, Sub};
use traits::Empty;
use num::traits::One;
use std::iter::Chain;

#[derive(Clone, Debug)]
pub struct Concat<'a, I, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Hash + PartialOrd + Eq
{
    pub concat_data: I,
    pub concat_other: Vec<S>,
    pub axis: UtahAxis,
}




impl<'a, I, T, S> Concat<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Hash + PartialOrd + Eq
{
    pub fn new(left_df: I,
               right_df: I,
               left_other: Vec<S>,
               axis: UtahAxis)
               -> Concat<'a, Chain<I, I>, T, S> {

        let it = left_df.chain(right_df);

        Concat {
            concat_data: it,
            concat_other: left_other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Concat<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Debug,
          S: Hash + PartialOrd + Eq + Debug
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.concat_data.next()
    }
}

#[derive(Clone)]
pub struct InnerJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub left: L,
    pub right: HashMap<S, RowView<'a, T>>,
    pub left_columns: Vec<S>,
    pub right_columns: Vec<S>,
}

impl<'a, L, T, S> InnerJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new<RI>(left: L, right: RI, left_columns: Vec<S>, right_columns: Vec<S>) -> Self
        where RI: Iterator<Item = (S, RowView<'a, T>)>
    {
        InnerJoin {
            left: left,
            right: right.collect(),
            left_columns: left_columns,
            right_columns: right_columns,
        }
    }
}



impl<'a, L, T, S> Iterator for InnerJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>, RowView<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.next() {
                Some((k, lv)) => {
                    let rv = self.right.get(&k);
                    match rv {
                        Some(v) => return Some((k, lv, *v)),
                        None => continue,
                    }
                }
                None => return None,
            }

        }
    }
}

#[derive(Clone)]
pub struct OuterJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    left: L,
    right: HashMap<S, RowView<'a, T>>,
    left_columns: Vec<S>,
    right_columns: Vec<S>,
}


impl<'a, L, T, S> OuterJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new<RI>(left: L, right: RI, left_columns: Vec<S>, right_columns: Vec<S>) -> Self
        where RI: Iterator<Item = (S, RowView<'a, T>)>
    {
        OuterJoin {
            left: left,
            right: right.collect(),
            left_columns: left_columns,
            right_columns: right_columns,
        }
    }
}


impl<'a, L, T, S> Iterator for OuterJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>, Option<RowView<'a, T>>);

    fn next(&mut self) -> Option<Self::Item> {

        match self.left.next() {
            Some((k, lv)) => {
                let rv = self.right.get(&k);
                match rv {
                    Some(v) => return Some((k, lv, Some(*v))),
                    None => Some((k, lv, None)),
                }

            }
            None => None,
        }

    }
}


impl<'a, L, T, S> ToDataFrame<'a, (S, RowView<'a, T>, RowView<'a, T>), T, S>
    for InnerJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
    T: 'a +Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T>+ One,
      S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug + From<String>
{
    fn as_df(self) -> DataFrame<T, S> {

        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), left_columns.len() + right_columns.len());


        for (i, j, k) in self {
            let p = j.iter().chain(k.iter()).map(|x| x.to_owned());
            c.extend(p);

            n.push(i.to_owned());
        }

        let columns: Vec<_> = left_columns.iter()
            .chain(right_columns.iter())
            .map(|x| x.to_owned())
            .collect();

        DataFrame {
            columns: columns,
            data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
            index: n,
        }


    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), left_columns.len() + right_columns.len());


        for (i, j, k) in self {
            let p = j.iter().chain(k.iter()).map(|x| x.to_owned());
            c.extend(p);

            n.push(i.to_owned());
        }


        Array::from_shape_vec(res_dim, c).unwrap()
    }

    fn as_array(self) -> Row<T> {
        let mut c = Vec::new();
        for (_, j, k) in self {
            let p = j.iter().chain(k.iter()).map(|x| x.to_owned());
            c.extend(p);
        }
        Array::from_vec(c)
    }
}


impl<'a, L,T,S> ToDataFrame<'a, (S, RowView<'a, T>, Option<RowView<'a, T>>), T, S>
    for OuterJoin<'a, L, T, S>
    where L: Iterator<Item = (S, RowView<'a, T>)> + Clone,
    T: 'a +Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T>+ One,
      S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug + From<String>
{
    fn as_df(self) -> DataFrame<T, S> {

        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), left_columns.len() + right_columns.len());

        let r = repeat(T::empty()).take(right_columns.len());
        for (i, j, k) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            match k {
                Some(z) => c.extend(z.iter().map(|x| x.to_owned())),
                None => c.extend(r.clone()),
            }


            n.push(i.to_owned());
        }

        let columns: Vec<_> = left_columns.iter()
            .chain(right_columns.iter())
            .map(|x| x.to_owned())
            .collect();

        DataFrame {
            columns: columns,
            data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
            index: n,
        }


    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), left_columns.len() + right_columns.len());

        let r = repeat(T::empty()).take(right_columns.len());
        for (i, j, k) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            match k {
                Some(z) => c.extend(z.iter().map(|x| x.to_owned())),
                None => c.extend(r.clone()),
            }


            n.push(i.to_owned());
        }

        Array::from_shape_vec(res_dim, c).unwrap()


    }

    fn as_array(self) -> Row<T> {
        let right_columns = self.right_columns.clone();
        let mut c = Vec::new();
        let r = repeat(T::empty()).take(right_columns.len());
        for (_, j, k) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            match k {
                Some(z) => c.extend(z.iter().map(|x| x.to_owned())),
                None => c.extend(r.clone()),
            }
        }
        Array::from_vec(c)
    }
}



impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Concat<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug+ Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug + From<String>
{
    fn as_df(self) -> DataFrame<T, S> {

        let s = self.clone();
        let columns = self.concat_other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), columns.len());

        for (i, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }
        println!("{:?}", c);
        DataFrame {
            columns: columns,
            data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
            index: n,
        }


    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let columns = self.concat_other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), columns.len());

        for (i, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }

        Array::from_shape_vec(res_dim, c).unwrap()


    }

    fn as_array(self) -> Row<T> {
        let mut c = Vec::new();

        for (_, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
        }
        Array::from_vec(c)
    }
}
