use types::*;
use std::iter::Iterator;
use ndarray::AxisIter;
use itertools::PutBack;
use std::slice::Iter;
use std::collections::HashMap;


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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
          F: Fn(&InnerType) -> B
{
    data: I,
    func: F,
    axis: UtahAxis,
}

impl<'a, I, F, B> MapDF<'a, I, F, B>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
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
    pub data: I,
    pub ind: Vec<OuterType>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}


impl<'a, I> Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, ind: Vec<OuterType>, other: Vec<OuterType>, axis: UtahAxis) -> Select<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
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
    pub data: I,
    pub ind: Vec<OuterType>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}


impl<'a, I> Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, ind: Vec<OuterType>, other: Vec<OuterType>, axis: UtahAxis) -> Remove<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
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

pub struct Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub new_data: PutBack<I>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}




impl<'a, I> Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I,
               name: OuterType,
               data: RowView<'a, InnerType>,
               other: Vec<OuterType>,
               axis: UtahAxis)
               -> Append<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
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
    type Item = (OuterType, Row<InnerType>, Row<InnerType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.next() {
                Some((k, lv)) => {
                    let rv = self.right.get(&k);
                    match rv {
                        Some(v) => return Some((k, lv.to_owned(), v.to_owned())),
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
    type Item = (OuterType, Row<InnerType>, Option<Row<InnerType>>);

    fn next(&mut self) -> Option<Self::Item> {

        match self.left.next() {
            Some((k, lv)) => {
                let rv = self.right.get(&k);
                match rv {
                    Some(v) => return Some((k, lv.to_owned(), Some(v.to_owned()))),
                    None => Some((k, lv.to_owned(), None)),
                }

            }
            None => None,
        }

    }
}
