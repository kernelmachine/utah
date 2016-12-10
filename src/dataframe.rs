use ndarray::Axis;
use std::collections::BTreeMap;
use join::*;
use error::*;
use types::*;
use std::string::ToString;
use std::iter::{Iterator, IntoIterator, Chain};
use ndarray::AxisIter;
use itertools::PutBack;
use std::slice::Iter;

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

pub struct Select<'a> {
    names: Iter<'a, OuterType>,
    data: AxisIter<'a, InnerType, usize>,
    ind: Vec<OuterType>,
}

pub struct Remove<'a> {
    names: Iter<'a, OuterType>,
    data: AxisIter<'a, InnerType, usize>,
    ind: Vec<OuterType>,
}

pub struct Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub new_data: PutBack<I>,
}


impl<'a> Iterator for Select<'a> {
    type Item = (OuterType, RowView<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.names.next() {

                Some(val) => {
                    match self.data.next() {

                        Some(dat) => {
                            if self.ind.contains(&val) {
                                return Some((val.clone(), dat));
                            } else {
                                continue;
                            }
                        }
                        None => return None,
                    }
                }
                None => return None,
            }


        }
    }
}

impl<'a> Iterator for Remove<'a> {
    type Item = (OuterType, RowView<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.names.next() {

                Some(val) => {
                    match self.data.next() {

                        Some(dat) => {
                            if !self.ind.contains(&val) {

                                return Some((val.clone(), dat));
                            } else {
                                continue;
                            }
                        }
                        None => return None,
                    }
                }
                None => return None,
            }


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


pub fn select<'a>(df: DataFrameIterator<'a>, ind: Vec<OuterType>) -> Select<'a> {

    Select {
        data: df.data,
        names: df.names,
        ind: ind,
    }
}

pub fn remove<'a>(df: DataFrameIterator<'a>, ind: Vec<OuterType>) -> Remove<'a> {

    Remove {
        data: df.data,
        names: df.names,
        ind: ind,
    }
}



pub fn append<'a>(df: DataFrameIterator<'a>,
                  name: OuterType,
                  data: RowView<'a, InnerType>)
                  -> Append<'a, DataFrameIterator<'a>> {
    let name = OuterType::from(name);
    let mut it = PutBack::new(df);
    it.put_back((name, data));
    Append { new_data: it }
}


pub fn join<'a>(this: DataFrameIterator<'a>,
                other: &DataFrameIterator<'a>)
                -> Chain<Select<'a>, Select<'a>> {
    let this_index: BTreeMap<OuterType, usize> =
        this.clone().names.enumerate().map(|(x, y)| (y.clone(), x)).collect();
    let other_index: BTreeMap<OuterType, usize> =
        other.clone().names.enumerate().map(|(x, y)| (y.clone(), x)).collect();
    let idxs: Vec<(OuterType, usize, Option<usize>)> =
        Join::new(JoinType::InnerJoin, this_index.into_iter(), other_index).collect();
    let i1: Vec<OuterType> =
        idxs.iter().filter(|x| x.2.is_some()).map(|&(ref x, _, _)| x.to_owned()).collect();
    let df: Select<'a> = this.select(i1.clone());
    let other_df: Select<'a> = other.clone().select(i1.clone());
    df.chain(other_df)
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


impl<'a> DataFrameIterator<'a> {
    pub fn select(self, names: Vec<OuterType>) -> Select<'a> {

        select(self, names)
    }


    pub fn remove(self, names: Vec<OuterType>) -> Remove<'a> {

        remove(self, names)

    }
    pub fn append(self,
                  name: OuterType,
                  data: RowView<'a, InnerType>)
                  -> Append<'a, DataFrameIterator<'a>> {
        append(self, name, data)

    }


    pub fn new(df: &'a DataFrame, axis: Axis) -> Self {
        match axis {
            Axis(0) => {
                DataFrameIterator {
                    names: df.index.iter(),
                    data: df.data.axis_iter(Axis(0)),
                }
            }
            Axis(1) => {
                DataFrameIterator {
                    names: df.columns.iter(),
                    data: df.data.axis_iter(Axis(1)),
                }
            }
            _ => panic!(),

        }


    }
}

impl DataFrame {
    pub fn new<T: Clone>(data: Matrix<T>) -> DataFrame
        where InnerType: From<T>
    {
        let data: Matrix<InnerType> = data.mapv(InnerType::from);

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
