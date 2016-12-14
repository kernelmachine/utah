use types::*;
use std::iter::{Iterator, Chain};
use ndarray::AxisIter;
use itertools::PutBack;
use std::slice::Iter;
use std::marker::Sized;
use std::collections::HashMap;
use aggregate::*;
use traits::*;

#[derive(Clone)]
pub struct DataFrameIterator<'a> {
    pub names: Iter<'a, OuterType>,
    pub data: AxisIter<'a, InnerType, usize>,
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          F: Fn(&InnerType) -> B
{
    data: I,
    func: F,
}

impl<'a, I, F, B> MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          F: Fn(&InnerType) -> B
{
    pub fn new(df: I, f: F) -> MapDF<'a, I, F, B>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        MapDF {
            data: df,
            func: f,
        }
    }
}

impl<'a, I, F, B> Iterator for MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
    ind: Vec<OuterType>,
}


impl<'a, I> Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, ind: Vec<OuterType>) -> Select<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Select {
            data: df,
            ind: ind,
        }
    }
}




impl<'a, I> Iterator for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
    ind: Vec<OuterType>,
}


impl<'a, I> Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, ind: Vec<OuterType>) -> Remove<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Remove {
            data: df,
            ind: ind,
        }
    }
}



impl<'a, I> Iterator for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub new_data: PutBack<I>,
}




impl<'a, I> Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, name: OuterType, data: RowView<'a, InnerType>) -> Append<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        let name = OuterType::from(name);
        let mut it = PutBack::new(df);
        it.put_back((name, data));
        Append { new_data: it }
    }
}


impl<'a, I> Iterator for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = (OuterType, RowView<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {
        self.new_data.next()
    }
}

pub struct InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    left: L,
    right: HashMap<OuterType, RowView<'a, InnerType>>,
}

impl<'a, L> InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new<RI>(left: L, right: RI) -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin {
            left: left,
            right: right.collect(),
        }
    }
}



impl<'a, L> Iterator for InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = (OuterType, RowView<'a, InnerType>, RowView<'a, InnerType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.next() {
                Some((k, lv)) => {
                    let rv = self.right.get(&k);
                    match rv {
                        Some(&v) => return Some((k, lv, v)),
                        None => continue,
                    }
                }
                None => return None,
            }

        }
    }
}


pub struct OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    left: L,
    right: HashMap<OuterType, RowView<'a, InnerType>>,
}


impl<'a, L> OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new<RI>(left: L, right: RI) -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin {
            left: left,
            right: right.collect(),
        }
    }
}


impl<'a, L> Iterator for OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = (OuterType, RowView<'a, InnerType>, Option<RowView<'a, InnerType>>);

    fn next(&mut self) -> Option<Self::Item> {

        match self.left.next() {
            Some((k, lv)) => {
                let rv = self.right.get(&k);
                match rv {
                    Some(&v) => return Some((k, lv, Some(v))),
                    None => Some((k, lv, None)),
                }

            }
            None => None,
        }

    }
}



impl<'a> DFIter<'a> for DataFrameIterator<'a> {
    type DFItem = (OuterType, RowView<'a, InnerType>);

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let name = OuterType::from(name);
        Append::new(self, name, data)

    }

    fn concat<I>(self, other: I) -> Chain<Self, I>
        where I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }



    fn inner_left_join<I>(self, other: I) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }

    fn outer_left_join<I>(self, other: I) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }

    fn inner_right_join<I>(self, other: I) -> InnerJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(other, self)
    }

    fn outer_right_join<I>(self, other: I) -> OuterJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        MapDF::new(self, f)
    }
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}

impl<'a, I> DFIter<'a> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type DFItem = (OuterType, RowView<'a, InnerType>);
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let name = OuterType::from(name);
        Append::new(self, name, data)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        MapDF::new(self, f)
    }

    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}

impl<'a, I> DFIter<'a> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type DFItem = (OuterType, RowView<'a, InnerType>);
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>
    {
        let name = OuterType::from(name);
        Append::new(self, name, data)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        MapDF::new(self, f)
    }

    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}

impl<'a, I> DFIter<'a> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type DFItem = (OuterType, RowView<'a, InnerType>);
    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>
    {
        let name = OuterType::from(name);
        Append::new(self, name, data)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        MapDF::new(self, f)
    }

    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}
