//! Utah transform combinators.

use util::types::*;
use std::iter::Iterator;
use itertools::PutBack;
use ndarray::Array;
use combinators::aggregate::*;
use util::traits::*;
use dataframe::*;
use std::fmt::Debug;
use util::error::*;


#[derive(Clone, Debug)]
pub struct Select<'a, I, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    pub data: I,
    pub ind: Vec<S>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, I, T, S> Select<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    pub fn new(df: I, ind: Vec<S>, other: Vec<S>, axis: UtahAxis) -> Select<'a, I, T, S> {

        Select {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}




impl<'a, I, T, S> Iterator for Select<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    type Item = (S, RowView<'a, T>);
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
pub struct Remove<'a, I, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    pub data: I,
    pub ind: Vec<S>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, I, T, S> Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    pub fn new(df: I, ind: Vec<S>, other: Vec<S>, axis: UtahAxis) -> Remove<'a, I, T, S> {

        Remove {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}



impl<'a, I, T, S> Iterator for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          S: Identifier
{
    type Item = (S, RowView<'a, T>);
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
pub struct Append<'a, I, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    pub new_data: PutBack<I>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}




impl<'a, I, T, S> Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    pub fn new(df: I,
               name: S,
               data: RowView<'a, T>,
               other: Vec<S>,
               axis: UtahAxis)
               -> Append<'a, I, T, S> {
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
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    type Item = (S, RowView<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.new_data.next()
    }
}






impl<'a, T, S> Aggregate<'a, T, S> for DataFrameIterator<'a, T, S>
    where T: Num + 'a,
          S: Identifier
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
}


impl<'a, T, S> Transform<'a, T, S> for DataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T, S>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<S> = names.iter()
            .map(|x| S::from(*x))
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
            .map(|x| S::from(*x))
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
}






impl<'a, I, T, S> Aggregate<'a, T, S> for Select<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num + 'a,
          S: Identifier
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
}

impl<'a, I, T, S> Transform<'a, T, S> for Select<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num + Clone + Debug,
          S: Identifier + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T, S>
        where S: From<&'a U>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| (S::from(*x)))
            .collect();
        Select::new(self, names, other.clone(), axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(*x))
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
}





impl<'a, I, T, S> Aggregate<'a, T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
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
}

impl<'a, I, T, S> Transform<'a, T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(*x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(*x))
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
}



impl<'a, I, T, S> Aggregate<'a, T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
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
}
impl<'a, I, T, S> Transform<'a, T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| S::from(*x))
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
            .map(|x| S::from(*x))
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
}

impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Remove<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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



impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Append<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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


impl<'a, I, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for Select<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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


impl<'a, T, S> ToDataFrame<'a, (S, RowView<'a, T>), T, S> for DataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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
