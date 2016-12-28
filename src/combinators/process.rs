//! Utah process combinators.

use util::types::*;
use std::iter::Iterator;
use dataframe::{DataFrame, MutableDataFrame, MutableDataFrameIterator};
use util::traits::*;
use ndarray::Array;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Div};
use util::error::*;


#[derive(Clone, Debug)]
pub struct MapDF<'a, T: 'a, S, I, F>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          F: Fn(T) -> T,
          S: Identifier
{
    data: I,
    func: F,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, T, S, I, F> MapDF<'a, T, S, I, F>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          F: Fn(T) -> T,
          S: Identifier
{
    pub fn new(df: I, f: F, other: Vec<S>, axis: UtahAxis) -> MapDF<'a, T, S, I, F> {

        MapDF {
            data: df,
            func: f,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, T, S, I, F> Iterator for MapDF<'a, T, S, I, F>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          F: Fn(T) -> T,
          S: Identifier,
          T: Clone
{
    type Item = (S, RowViewMut<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((val, mut dat)) => {
                dat.mapv_inplace(&self.func);
                return Some((val, dat));
            }
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

impl<'a, I, T, S> Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          T: Num,
          S: Identifier
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
          T: Num,
          S: Identifier
{
    type Item = (S, RowViewMut<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((val, mut dat)) => {
                match self.strategy {
                    ImputeStrategy::Mean => {

                        let size = dat.iter()
                            .filter(|&x| !(*x).is_empty())
                            .fold(T::zero(), |acc, _| acc + T::one());
                        let nonempty = dat.iter()
                            .filter(|&x| !(*x).is_empty())
                            .fold(T::zero(), |acc, x| acc + x.clone());
                        let mean = nonempty / size;

                        dat.mapv_inplace(|x| {
                            if x.is_empty() {
                                return mean.clone();
                            } else {
                                return x.clone();
                            }
                        });

                        Some((val, dat))
                    }

                }
            }
        }
    }
}


impl<'a, I, T, S, F> Process<'a, T, S, F> for MapDF<'a, T, S, I, F>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          T: Num,
          F: Fn(T) -> T,
          S: Identifier
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {

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

    fn mapdf(self, f: F) -> MapDF<'a, T, S, Self, F>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}


impl<'a, I, T, S, F> Process<'a, T, S, F> for Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          T: Num,
          S: Identifier,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {

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

    fn mapdf(self, f: F) -> MapDF<'a, T, S, Self, F> {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}

impl<'a, T, S, F> Process<'a, T, S, F> for MutableDataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
    {

        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>
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

    fn mapdf(self, f: F) -> MapDF<'a, T, S, Self, F> {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}

impl<'a, T, S> ToDataFrame<'a, (S, RowViewMut<'a, T>), T, S> for MutableDataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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


        let d = Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone()));
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone())))


    }

    fn as_array(self) -> Result<Row<T>> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j);
        }
        Ok(Array::from_vec(c).map(|x| ((*x).clone())))
    }
}

impl<'a, I, T, S> ToDataFrame<'a, (S, RowViewMut<'a, T>), T, S> for Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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


        let d = Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone()));
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }

        }
    }

    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone())))
    }

    fn as_array(self) -> Result<Row<T>> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j);
        }
        Ok(Array::from_vec(c).map(|x| ((*x).clone())))
    }
}



impl<'a, I, T, S, F> ToDataFrame<'a, (S, RowViewMut<'a, T>), T, S> for MapDF<'a, T, S, I, F>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
          T: Num,
          S: Identifier,
          F: Fn(T) -> T
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
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


        let d = Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone()));
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&n[..])?;
                Ok(df)
            }

        }
    }

    fn as_matrix(self) -> Result<Matrix<T>> {
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

        Ok(Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone())))
    }

    fn as_array(self) -> Result<Row<T>> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j);
        }
        Ok(Array::from_vec(c).map(|x| ((*x).clone())))
    }
}
