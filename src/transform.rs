
use types::*;
use std::iter::Iterator;
use ndarray::AxisIter;
use itertools::PutBack;
use std::slice::Iter;
use ndarray::Array;
use aggregate::*;
use traits::*;
use std::iter::Chain;
use dataframe::*;

#[derive(Clone)]
pub struct DataFrameIterator<'a> {
    pub names: Iter<'a, OuterType>,
    pub data: AxisIter<'a, InnerType, usize>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}




impl<'a> Iterator for DataFrameIterator<'a> {
    type Item = (OuterType, RowView<'a, InnerType>);
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
pub struct MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
          F: Fn(&InnerType) -> B
{
    data: I,
    func: F,
    axis: UtahAxis,
}

impl<'a, I, F, B> MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
          F: Fn(&InnerType) -> B
{
    pub fn new(df: I, f: F, axis: UtahAxis) -> MapDF<'a, I, F, B>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        MapDF {
            data: df,
            func: f,
            axis: axis,
        }
    }
}

impl<'a, I, F, B> Iterator for MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
          F: Fn(&InnerType) -> B,
          B: 'a
{
    type Item = (OuterType, Row<B>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((val, dat)) => return Some((val, dat.map(&self.func))),
        }
    }
}





#[derive(Clone)]

pub struct Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub data: I,
    pub ind: Vec<OuterType>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}


impl<'a, I> Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub fn new(df: I, ind: Vec<OuterType>, other: Vec<OuterType>, axis: UtahAxis) -> Select<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {

        Select {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}




impl<'a, I> Iterator for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    type Item = (OuterType, RowView<'a, InnerType>);
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
pub struct Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub data: I,
    pub ind: Vec<OuterType>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}


impl<'a, I> Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub fn new(df: I, ind: Vec<OuterType>, other: Vec<OuterType>, axis: UtahAxis) -> Remove<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {

        Remove {
            data: df,
            ind: ind,
            other: other,
            axis: axis,
        }
    }
}



impl<'a, I> Iterator for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    type Item = (OuterType, RowView<'a, InnerType>);
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
pub struct Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub new_data: PutBack<I>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}




impl<'a, I> Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub fn new(df: I,
               name: OuterType,
               data: RowView<'a, InnerType>,
               other: Vec<OuterType>,
               axis: UtahAxis)
               -> Append<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let name = OuterType::from(name);
        let mut it = PutBack::new(df);
        it.put_back((name, data));
        Append {
            new_data: it,
            other: other,
            axis: axis,
        }
    }
}


impl<'a, I> Iterator for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    type Item = (OuterType, RowView<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {
        self.new_data.next()
    }
}



impl<'a> Aggregate<'a> for DataFrameIterator<'a> {
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}


impl<'a> Transform<'a> for DataFrameIterator<'a> {
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }

    fn concat<I>(self, other: I) -> Chain<Self, I>
        where I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }




    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}






impl<'a, I> Aggregate<'a> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}

impl<'a, I> Transform<'a> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other.clone(), axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other.clone(), axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }


    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}





impl<'a, I> Aggregate<'a> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}

impl<'a, I> Transform<'a> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }



    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}



impl<'a, I> Aggregate<'a> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Sum::new(self, other, axis)
    }

    fn maxdf(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Max::new(self, other, axis)
    }

    fn mindf(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Min::new(self, other, axis)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Mean::new(self, other, axis)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Stdev::new(self, other, axis)
    }
}
impl<'a, I> Transform<'a> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }


    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B,
              Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}

impl<'a, I> ToDataFrame<'a, (OuterType, RowView<'a, InnerType>)> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
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
}



impl<'a, I> ToDataFrame<'a, (OuterType, RowView<'a, InnerType>)> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
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
}


impl<'a, I> ToDataFrame<'a, (OuterType, RowView<'a, InnerType>)> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
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
}


impl<'a> ToDataFrame<'a, (OuterType, RowView<'a, InnerType>)> for DataFrameIterator<'a> {
    fn to_df(self) -> DataFrame {
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
}
