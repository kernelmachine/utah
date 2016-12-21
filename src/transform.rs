
use types::*;
use std::iter::Iterator;
use itertools::PutBack;
use std::slice::Iter;
use ndarray::Array;
use aggregate::*;
use traits::*;
use std::iter::Chain;
use dataframe::*;
use std::hash::Hash;
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};
use num::traits::One;

#[derive(Clone)]
pub struct DataFrameIterator<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug + 'a
{
    pub names: Iter<'a, S>,
    pub data: I,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}




impl<'a, I, T, S> Iterator for DataFrameIterator<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
          T: Clone + Debug,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.names.next() {
            Some(val) => {
                match self.data.next() {
                    Some(dat) => Some((val.clone(), dat)),
                    None => None,
                }
            }
            None => None,
        }
    }
}

#[derive(Clone)]
pub struct MapDF<'a, T, S, I, F, B>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          F: Fn(&T) -> B,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    func: F,
    axis: UtahAxis,
}

impl<'a, T, S, I, F, B> MapDF<'a, T, S, I, F, B>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          F: Fn(&T) -> B,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, f: F, axis: UtahAxis) -> MapDF<'a, T, S, I, F, B>
        where I: Iterator<Item = (S, RowView<'a, T>)>
    {

        MapDF {
            data: df,
            func: f,
            axis: axis,
        }
    }
}

impl<'a, T, S, I, F, B> Iterator for MapDF<'a, T, S, I, F, B>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          F: Fn(&T) -> B,
          B: 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, Row<B>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((val, dat)) => return Some((val, dat.map(&self.func))),
        }
    }
}





#[derive(Clone)]

pub struct Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub data: I,
    pub ind: Vec<S>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, T, S, I> Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, ind: Vec<S>, other: Vec<S>, axis: UtahAxis) -> Select<'a, T, S, I>
        where I: Iterator<Item = (S, RowView<'a, T>)> + Clone
    {

        Select {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}




impl<'a, I, T, S> Iterator for Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.data.next() {
                Some((val, dat)) => {
                    if self.ind.contains(&val) {
                        return Some((val.clone(), dat));
                    } else {
                        continue;
                    }
                }
                None => return None,
            }


        }
    }
}

#[derive(Clone)]
pub struct Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub data: I,
    pub ind: Vec<S>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, I, T, S> Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, ind: Vec<S>, other: Vec<S>, axis: UtahAxis) -> Remove<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)> + Clone
    {

        Remove {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}



impl<'a, I, T, S> Iterator for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.data.next() {

                Some((val, dat)) => {
                    if !self.ind.contains(&val) {
                        return Some((val.clone(), dat));
                    } else {
                        continue;
                    }
                }
                None => return None,
            }
        }
    }
}

#[derive(Clone)]
pub struct Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub new_data: PutBack<I>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}




impl<'a, I, T, S> Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I,
               name: S,
               data: RowView<'a, T>,
               other: Vec<S>,
               axis: UtahAxis)
               -> Append<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let name = S::from(name);
        let mut it = PutBack::new(df);
        it.put_back((name, data));
        Append {
            new_data: it,
            other: other,
            axis: axis,
        }
    }
}


impl<'a, I, T, S> Iterator for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.new_data.next()
    }
}



impl<'a, I, T, S> Aggregate<'a, T, S> for DataFrameIterator<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn sumdf(self) -> Sum<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}


impl<'a, I, T, S> Transform<'a, T, S> for DataFrameIterator<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, T, S, Self>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<S> = names.iter()
            .map(|x| S::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<S> = names.iter()
            .map(|x| S::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: RowView<'a, T>) -> Append<'a, Self, T, S>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = S::from(name);
        Append::new(self, name, data, other, axis)

    }

    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        self.chain(other)
    }




    fn mapdf<F, B>(self, f: F) -> MapDF<'a, T, S, Self, F, B>
        where F: Fn(&T) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}






impl<'a, I, T, S> Aggregate<'a, T, S> for Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn sumdf(self) -> Sum<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}

impl<'a, I, T, S> Transform<'a, T, S> for Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, T, S, Self>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| (S::from(x)))
            .collect();
        Select::new(self, names, other.clone(), axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(x))
            .collect();
        Remove::new(self, names, other.clone(), axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: RowView<'a, T>) -> Append<'a, Self, T, S>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = S::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        self.chain(other)
    }


    fn mapdf<F, B>(self, f: F) -> MapDF<'a, T, S, Self, F, B>
        where F: Fn(&T) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}





impl<'a, I, T, S> Aggregate<'a, T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn sumdf(self) -> Sum<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}

impl<'a, I, T, S> Transform<'a, T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, T, S, Self>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: RowView<'a, T>) -> Append<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = S::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        self.chain(other)
    }



    fn mapdf<F, B>(self, f: F) -> MapDF<'a, T, S, Self, F, B>
        where F: Fn(&T) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}



impl<'a, I, T, S> Aggregate<'a, T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn sumdf(self) -> Sum<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}
impl<'a, I, T, S> Transform<'a, T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, T, S, Self>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: RowView<'a, T>) -> Append<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = S::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (S, RowView<'a, T>)>
    {
        self.chain(other)
    }


    fn mapdf<F, B>(self, f: F) -> MapDF<'a, T, S, Self, F, B>
        where F: Fn(&T) -> B,
              Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}

impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Copy +Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
    fn as_df(self) -> DataFrame<T, S> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

        for (i, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }

        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();

        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

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



impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T:Copy + Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
    fn as_df(self) -> DataFrame<T, S> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

        for (i, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }


        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();

        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

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


impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Select<'a, T, S, I>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Copy +Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
    fn as_df(self) -> DataFrame<T, S> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

        for (i, j) in self {

            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }

        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();

        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

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


impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for DataFrameIterator<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
            T: Copy +Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
    fn as_df(self) -> DataFrame<T, S> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

        for (i, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            n.push(i.to_owned());
        }

        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
    fn as_matrix(self) -> Matrix<T> {
        let s = self.clone();
        let other = self.other.clone();
        let axis = self.axis.clone();

        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = match axis {
            UtahAxis::Row => (s.fold(0, |acc, _| acc + 1), other.len()),
            UtahAxis::Column => (other.len(), s.fold(0, |acc, _| acc + 1)),
        };

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
