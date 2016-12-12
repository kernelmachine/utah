use ndarray::Axis;
use error::*;
use types::*;
use std::string::ToString;
use std::iter::{Iterator, Chain};
use ndarray::AxisIter;
use itertools::PutBack;
use std::slice::Iter;
use std::marker::Sized;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    pub columns: Vec<OuterType>,
    pub data: Matrix<InnerType>,
    pub index: Vec<OuterType>,
}

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
pub struct Dot<'a, I, J>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          J: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    this_data: I,
    other_data: J,
}

impl<'a, I, J> Dot<'a, I, J>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          J: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: J) -> Dot<'a, I, J>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Dot {
            this_data: df,
            other_data: other,
        }
    }
}


impl<'a, I, J> Iterator for Dot<'a, I, J>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          J: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.this_data.next() {

            None => return None,
            Some((_, dat)) => {
                match self.other_data.next() {
                    Some((_, other_dat)) => {
                        return Some((0..dat.len()).fold(InnerType::Int32(0), |x, y| {
                            x +
                            dat.get(y).unwrap().to_owned() * other_dat.get(y).unwrap().to_owned()
                        }))
                    }
                    None => return None,
                }
            }

        }
    }
}

#[derive(Clone)]
pub struct Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Sum<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Sum { data: df }
    }
}

impl<'a, I> Iterator for Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => {
                return Some((0..dat.len()).fold(dat.get(0).unwrap().to_owned(),
                                                |x, y| x + dat.get(y).unwrap().to_owned()))
            }
        }
    }
}

#[derive(Clone)]
pub struct Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Mean<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Mean { data: df }
    }
}

impl<'a, I> Iterator for Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                match dat.uget(0) {
                    &InnerType::Float(_) => return Some(sum / InnerType::Float(size as f64)),
                    &InnerType::Int32(_) => return Some(sum / InnerType::Int32(size as i32)),
                    &InnerType::Int64(_) => return Some(sum / InnerType::Int64(size as i64)),
                    _ => return Some(InnerType::Empty),
                }

            },
        }
    }
}


#[derive(Clone)]
pub struct Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Max<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Max { data: df }
    }
}

impl<'a, I> Iterator for Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().max().map(|x| x.to_owned()),
        }



    }
}


#[derive(Clone)]
pub struct Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Min<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Min { data: df }
    }
}

impl<'a, I> Iterator for Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().min().map(|x| x.to_owned()),
        }



    }
}

#[derive(Clone)]
pub struct Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Stdev<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Stdev { data: df }
    }
}

impl<'a, I> Iterator for Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                let mean = match dat.uget(0) {
                    &InnerType::Float(_) => sum / InnerType::Float(size as f64),
                    &InnerType::Int32(_) => sum / InnerType::Int32(size as i32),
                    &InnerType::Int64(_) => sum / InnerType::Int64(size as i64),
                    _ => InnerType::Empty,
                };

                let stdev = (0..size).fold(dat.uget(0).to_owned(), |x, y| {
                    x +
                    (dat.uget(y).to_owned() - mean.to_owned()) *
                    (dat.uget(y).to_owned() - mean.to_owned())
                });


                Some(stdev)


            },
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




pub trait DFIter<'a> {
    type DFItem;

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn concat<I>(self, other: I) -> Chain<Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn inner_left_join<I>(self, other: I) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
    fn outer_left_join<I>(self, other: I) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
    fn inner_right_join<I>(self, other: I) -> InnerJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
    fn outer_right_join<I>(self, other: I) -> OuterJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              F: Fn(&InnerType) -> B;
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn dot<I>(self, other: I) -> Dot<'a, Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
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

    fn dot<I>(self, other: I) -> Dot<'a, Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Dot::new(self, other)
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

    fn dot<J>(self, other: J) -> Dot<'a, Self, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Dot::new(self, other)
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


    fn dot<J>(self, other: J) -> Dot<'a, Self, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Dot::new(self, other)
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

    fn dot<J>(self, other: J) -> Dot<'a, Self, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Dot::new(self, other)
    }
}






impl DataFrame {
    pub fn new<T: Clone>(data: Matrix<T>) -> DataFrame
        where InnerType: From<T>
    {
        let data: Matrix<InnerType> = data.mapv(InnerType::from);
        let data: Matrix<InnerType> = data.mapv_into(|x| {
            match x {
                InnerType::Float(y) => {
                    if y.is_nan() {
                        return InnerType::Empty;
                    } else {
                        return x;
                    }
                }
                _ => return x,
            }

        });
        let columns: Vec<OuterType> = (0..data.shape()[1])
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        let index: Vec<OuterType> = (0..data.shape()[0])
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }


    pub fn columns<'a, T>(mut self, columns: &'a [T]) -> Result<DataFrame>
        where OuterType: From<&'a T>
    {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Ok(self)
    }

    pub fn index<'a, T>(mut self, index: &'a [T]) -> Result<DataFrame>
        where OuterType: From<&'a T>
    {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Ok(self)
    }

    pub fn df_iter<'a>(&'a self, axis: Axis) -> DataFrameIterator<'a> {
        match axis {
            Axis(0) => {
                DataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter(Axis(0)),
                }
            }
            Axis(1) => {
                DataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter(Axis(1)),
                }
            }
            _ => panic!(),

        }
    }
    pub fn select<'a, T>(&'a self, names: &'a [T], axis: Axis) -> Select<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            Axis(0) => Select::new(self.df_iter(Axis(0)), names),
            Axis(1) => Select::new(self.df_iter(Axis(1)), names),
            _ => panic!(),

        }
    }


    pub fn remove<'a, T>(&'a self, names: &'a [T], axis: Axis) -> Remove<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            Axis(0) => Remove::new(self.df_iter(Axis(0)), names),
            Axis(1) => Remove::new(self.df_iter(Axis(1)), names),
            _ => panic!(),

        }
    }

    pub fn append<'a, T>(&'a self,
                         name: &'a T,
                         data: RowView<'a, InnerType>,
                         axis: Axis)
                         -> Append<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let name = OuterType::from(name);
        match axis {
            Axis(0) => Append::new(self.df_iter(Axis(0)), name, data),
            Axis(1) => Append::new(self.df_iter(Axis(1)), name, data),
            _ => panic!(),

        }
    }

    pub fn inner_left_join<'a>(&'a self,
                               other: &'a DataFrame,
                               axis: Axis)
                               -> InnerJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => InnerJoin::new(self.df_iter(Axis(0)), other.df_iter(Axis(0))),
            Axis(1) => InnerJoin::new(self.df_iter(Axis(1)), other.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
    pub fn outer_left_join<'a>(&'a self,
                               other: &'a DataFrame,
                               axis: Axis)
                               -> OuterJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => OuterJoin::new(self.df_iter(Axis(0)), other.df_iter(Axis(0))),
            Axis(1) => OuterJoin::new(self.df_iter(Axis(1)), other.df_iter(Axis(1))),
            _ => panic!(),

        }

    }
    pub fn inner_right_join<'a>(&'a self,
                                other: &'a DataFrame,
                                axis: Axis)
                                -> InnerJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => InnerJoin::new(other.df_iter(Axis(0)), self.df_iter(Axis(0))),
            Axis(1) => InnerJoin::new(other.df_iter(Axis(1)), self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
    pub fn outer_right_join<'a>(&'a self,
                                other: &'a DataFrame,
                                axis: Axis)
                                -> OuterJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => OuterJoin::new(other.df_iter(Axis(0)), self.df_iter(Axis(0))),
            Axis(1) => OuterJoin::new(other.df_iter(Axis(1)), self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
    pub fn dot<'a>(&'a self,
                   other: &'a DataFrame)
                   -> Chain<Dot<'a, DataFrameIterator<'a>, DataFrameIterator<'a>>,
                            Dot<'a, DataFrameIterator<'a>, DataFrameIterator<'a>>> {

        Dot::new(self.df_iter(Axis(0)), other.df_iter(Axis(1)))
            .chain(Dot::new(self.df_iter(Axis(1)), other.df_iter(Axis(0))))
    }


    pub fn sumdf<'a>(&'a self, axis: Axis) -> Sum<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Sum::new(self.df_iter(Axis(0))),
            Axis(1) => Sum::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn map<'a, F, B>(&'a self, f: F, axis: Axis) -> MapDF<'a, DataFrameIterator<'a>, F, B>
        where F: Fn(&InnerType) -> B
    {

        match axis {
            Axis(0) => MapDF::new(self.df_iter(Axis(0)), f),
            Axis(1) => MapDF::new(self.df_iter(Axis(1)), f),
            _ => panic!(),

        }
    }

    pub fn mean<'a>(&'a self, axis: Axis) -> Mean<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Mean::new(self.df_iter(Axis(0))),
            Axis(1) => Mean::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn max<'a>(&'a self, axis: Axis) -> Max<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Max::new(self.df_iter(Axis(0))),
            Axis(1) => Max::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn min<'a>(&'a self, axis: Axis) -> Min<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Min::new(self.df_iter(Axis(0))),
            Axis(1) => Min::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn stdev<'a>(&'a self, axis: Axis) -> Stdev<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Stdev::new(self.df_iter(Axis(0))),
            Axis(1) => Stdev::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
}



// To implement....?
// // parallelized join
// // parallelized concatenation
// // parallelized frequency counts
// // index dataframe?
// // sample rows
// // find/select
// // sort
// // statistics (mean, median, stdev)
// // print
//
// // statistics (mean, median, stdev)
// // print
