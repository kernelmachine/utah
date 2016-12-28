//! Utah transform combinators.

use util::types::*;
use std::iter::Iterator;
use itertools::{put_back, PutBack};
use ndarray::{Array, ArrayView1};
use combinators::aggregate::*;
use util::traits::*;
use dataframe::*;
use std::fmt::Debug;
use util::error::*;


#[derive(Clone, Debug)]
pub struct Select<'a, I, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    pub data: I,
    pub ind: Vec<String>,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}


impl<'a, I, T> Select<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    pub fn new(df: I, ind: Vec<String>, other: Vec<String>, axis: UtahAxis) -> Select<'a, I, T> {

        Select {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}




impl<'a, I, T> Iterator for Select<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    type Item = (String, ArrayView1<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.data.next() {
                Some((val, dat)) => {
                    if self.ind.contains(&val) {
                        return Some((val, dat));
                    } else {
                        continue;
                    }
                }
                None => return None,
            }


        }
    }
}

#[derive(Clone, Debug)]
pub struct Remove<'a, I, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    pub data: I,
    pub ind: Vec<String>,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}


impl<'a, I, T> Remove<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    pub fn new(df: I, ind: Vec<String>, other: Vec<String>, axis: UtahAxis) -> Remove<'a, I, T> {

        Remove {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}



impl<'a, I, T> Iterator for Remove<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>
{
    type Item = (String, ArrayView1<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.data.next() {
                Some((val, dat)) => {
                    if !self.ind.contains(&val) {
                        return Some((val, dat));
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
pub struct Append<'a, I, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: UtahNum
{
    pub new_data: PutBack<I>,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}




impl<'a, I, T> Append<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: UtahNum
{
    pub fn new(df: I,
               name: String,
               data: ArrayView1<'a, T>,
               other: Vec<String>,
               axis: UtahAxis)
               -> Append<'a, I, T> {
        let mut it = put_back(df);
        it.put_back((name, data));
        Append {
            new_data: it,
            other: other,
            axis: axis,
        }
    }
}



impl<'a, I, T> Iterator for Append<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: UtahNum
{
    type Item = (String, ArrayView1<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.new_data.next()
    }
}






impl<'a, T> Aggregate<'a, T> for DataFrameIterator<'a, T>
    where T: UtahNum + 'a
{
    fn sumdf(self) -> Sum<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }
}


impl<'a, T> Transform<'a, T> for DataFrameIterator<'a, T>
    where T: UtahNum
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T>
        where String: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<String> = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T>
        where String: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<String> = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: ArrayView1<'a, T>) -> Append<'a, Self, T>
        where String: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = String::from(name);
        Append::new(self, name, data, other, axis)

    }
}






impl<'a, I, T> Aggregate<'a, T> for Select<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum + 'a
{
    fn sumdf(self) -> Sum<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }
}

impl<'a, I, T> Transform<'a, T> for Select<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T>
        where String: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| (String::from(*x)))
            .collect();
        Select::new(self, names, other.clone(), axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T>
        where String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Remove::new(self, names, other.clone(), axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: ArrayView1<'a, T>) -> Append<'a, Self, T>
        where String: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = String::from(name);
        Append::new(self, name, data, other, axis)

    }
}





impl<'a, I, T> Aggregate<'a, T> for Remove<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn sumdf(self) -> Sum<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }
}

impl<'a, I, T> Transform<'a, T> for Remove<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T>
        where String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T>
        where String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: ArrayView1<'a, T>) -> Append<'a, Self, T>
        where String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = String::from(name);
        Append::new(self, name, data, other, axis)

    }
}



impl<'a, I, T> Aggregate<'a, T> for Append<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn sumdf(self) -> Sum<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }
}
impl<'a, I, T> Transform<'a, T> for Append<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
              String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
              String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| String::from(*x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<U: ?Sized>(self, name: &'a U, data: ArrayView1<'a, T>) -> Append<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
              String: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = String::from(name);
        Append::new(self, name, data, other, axis)

    }
}

impl<'a, I, T> ToDataFrame<'a, (String, ArrayView1<'a, T>), T> for Remove<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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

        let d = Array::from_shape_vec(res_dim, c).unwrap();
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&n[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec(res_dim, c).unwrap())
    }

    fn as_array(self) -> Result<Row<T>> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
        }
        Ok(Array::from_vec(c))
    }
}



impl<'a, I, T> ToDataFrame<'a, (String, ArrayView1<'a, T>), T> for Append<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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


        let d = Array::from_shape_vec(res_dim, c).unwrap();
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&n[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec(res_dim, c).unwrap())
    }

    fn as_array(self) -> Result<Row<T>> {

        let mut c = Vec::new();


        for (_, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
        }

        Ok(Array::from_vec(c))
    }
}


impl<'a, I, T> ToDataFrame<'a, (String, ArrayView1<'a, T>), T> for Select<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + Clone,
          T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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

        let d = Array::from_shape_vec(res_dim, c).unwrap();
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&n[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec(res_dim, c).unwrap())
    }

    fn as_array(self) -> Result<Row<T>> {

        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
        }
        Ok(Array::from_vec(c))
    }
}


impl<'a, T> ToDataFrame<'a, (String, ArrayView1<'a, T>), T> for DataFrameIterator<'a, T>
    where T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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

        let d = Array::from_shape_vec(res_dim, c).unwrap();
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&n[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec(res_dim, c).unwrap())

    }

    fn as_array(self) -> Result<Row<T>> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
        }
        Ok(Array::from_vec(c))
    }
}
