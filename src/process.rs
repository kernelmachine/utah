use types::*;
use std::iter::Iterator;
use ndarray::AxisIterMut;
use std::slice::Iter;
use dataframe::{DataFrame, MutableDataFrame};
use traits::*;
use ndarray::Array;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Div};

pub struct MutableDataFrameIterator<'a, T, S>
where T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
      S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a {
    pub names: Iter<'a, S>,
    pub data: AxisIterMut<'a, T, usize>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, T, S> Iterator for MutableDataFrameIterator<'a, T, S>
where T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
      S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a{
    type Item = (S, RowViewMut<'a, T>);


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
pub struct Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)> + 'a,
        T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub data: I,
    pub strategy: ImputeStrategy,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}

impl<'a, I, T, S> Impute<'a, I,T,S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
            T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, s: ImputeStrategy, other: Vec<S>, axis: UtahAxis) -> Impute<'a, I, T, S>
        where I: Iterator<Item = (S, RowViewMut<'a, T>)>
    {

        Impute {
            data: df,
            strategy: s,
            axis: axis,
            other: other,
        }
    }
}

impl<'a, I, T, S> Iterator for Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
    T: Clone + Debug + 'a + Ord + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T>,
  S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowViewMut<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        let emp : T = T::empty();
        match self.data.next() {

            None => return None,
            Some((val, mut dat)) => {
                match self.strategy {
                    ImputeStrategy::Mean => unsafe {
                        let size = dat.len();

                        let first_element = dat.uget(0).to_owned();
                        let mean = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                        // let mean = match dat.uget(0) {
                        //     &InnerType::Float(_) => sum / InnerType::Float(size as f64),
                        //     &InnerType::Int32(_) => sum / InnerType::Int32(size as i32),
                        //     &InnerType::Int64(_) => sum / InnerType::Int64(size as i64),
                        //     _ => InnerType::Empty,
                        // };
                        dat.mapv_inplace(|x| {
                            if x == emp {
                                mean.to_owned()
                            }
                            else{
                                x.to_owned()
                            }

                        });
                        Some((val, dat))
                    },

                    ImputeStrategy::Mode => {
                        let max = dat.iter().max().map(|x| x.to_owned()).unwrap();
                        dat.mapv_inplace(|x| {
                            if x == emp {
                                max.to_owned()
                            }
                            else{
                                x.to_owned()
                            }

                        });
                        return Some((val, dat));
                    }
                }
            }
        }
    }
}


impl<'a, I> Process<'a, InnerType, OuterType> for Impute<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, InnerType, OuterType>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, InnerType, OuterType>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {

        // let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (other.len(), 0),
            UtahAxis::Column => (0, other.len()),
        };

        for (i, j) in self {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                MutableDataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: n,
                }
            }
            UtahAxis::Column => {
                MutableDataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: other,
                }
            }

        }

    }
}

impl<'a> Process<'a, InnerType, OuterType> for MutableDataFrameIterator<'a, InnerType, OuterType> {
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, InnerType, OuterType>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {

        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, InnerType, OuterType>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {
        // let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (other.len(), 0),
            UtahAxis::Column => (0, other.len()),
        };

        for (i, j) in self {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                MutableDataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: n,
                }
            }
            UtahAxis::Column => {
                MutableDataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: other,
                }
            }

        }
    }
}

impl<'a> ToDataFrame<'a, (OuterType, RowViewMut<'a, InnerType>), InnerType, OuterType> for MutableDataFrameIterator<'a, InnerType, OuterType>
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        self.to_mut_df().to_df()
    }
}

impl<'a, I> ToDataFrame<'a, (OuterType, RowViewMut<'a, InnerType>), InnerType, OuterType> for Impute<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        self.to_mut_df().to_df()
    }
}
