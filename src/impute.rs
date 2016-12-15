use types::*;
use std::iter::Iterator;
use ndarray::AxisIterMut;
use std::slice::Iter;


pub struct MutableDataFrameIterator<'a> {
    pub names: Iter<'a, OuterType>,
    pub data: AxisIterMut<'a, InnerType, usize>,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}


impl<'a> Iterator for MutableDataFrameIterator<'a> {
    type Item = (OuterType, RowViewMut<'a, InnerType>);
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
pub struct Impute<'a, I>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + 'a
{
    pub data: I,
    pub strategy: ImputeStrategy,
    pub other: Vec<OuterType>,
    pub axis: UtahAxis,
}

impl<'a, I> Impute<'a, I>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
{
    pub fn new(df: I, s: ImputeStrategy, other: Vec<OuterType>, axis: UtahAxis) -> Impute<'a, I>
        where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {

        Impute {
            data: df,
            strategy: s,
            axis: axis,
            other: other,
        }
    }
}

impl<'a, I> Iterator for Impute<'a, I>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
{
    type Item = (OuterType, RowViewMut<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {

        match self.data.next() {

            None => return None,
            Some((val, mut dat)) => {
                match self.strategy {
                    ImputeStrategy::Mean => unsafe {
                        let size = dat.len();
                        let first_element = dat.uget(0).to_owned();
                        let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                        let mean = match dat.uget(0) {
                            &InnerType::Float(_) => sum / InnerType::Float(size as f64),
                            &InnerType::Int32(_) => sum / InnerType::Int32(size as i32),
                            &InnerType::Int64(_) => sum / InnerType::Int64(size as i64),
                            _ => InnerType::Empty,
                        };
                        println!("{:?}", mean);
                        dat.mapv_inplace(|x| {
                            match x {
                                InnerType::Empty => mean.to_owned(),
                                _ => x.to_owned(),
                            }
                        });
                        Some((val, dat))
                    },
                    ImputeStrategy::Mode => {
                        let max = dat.iter().max().map(|x| x.to_owned()).unwrap();
                        dat.mapv_inplace(|x| {
                            match x {
                                InnerType::Empty => max.to_owned(),
                                _ => x.to_owned(),
                            }

                        });
                        return Some((val, dat));
                    }
                }
            }
        }
    }
}
