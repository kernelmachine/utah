//! Utah process combinators.

use util::types::*;
use std::iter::Iterator;
use dataframe::{DataFrame, MutableDataFrame, MutableDataFrameIterator};
use util::traits::*;
use ndarray::{ArrayViewMut1, Array};
use util::error::*;


#[derive(Clone, Debug)]
pub struct MapDF<'a, T: 'a, I, F>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          F: Fn(T) -> T
{
    data: I,
    func: F,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, T, I, F> MapDF<'a, T, I, F>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          F: Fn(T) -> T
{
    pub fn new(df: I, f: F, other: Vec<String>, axis: UtahAxis) -> MapDF<'a, T, I, F> {

        MapDF {
            data: df,
            func: f,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, T, I, F> Iterator for MapDF<'a, T, I, F>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          F: Fn(T) -> T,
          T: Clone
{
    type Item = (String, ArrayViewMut1<'a, T>);
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
pub struct Impute<'a, I, T: 'a>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)> + 'a,
          T: UtahNum
{
    pub data: I,
    pub strategy: ImputeStrategy,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}

impl<'a, I, T> Impute<'a, I, T>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum
{
    pub fn new(df: I, s: ImputeStrategy, other: Vec<String>, axis: UtahAxis) -> Impute<'a, I, T>
        where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>
    {

        Impute {
            data: df,
            strategy: s,
            axis: axis,
            other: other,
        }
    }
}

impl<'a, I, T> Iterator for Impute<'a, I, T>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum
{
    type Item = (String, ArrayViewMut1<'a, T>);
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


impl<'a, I, T, F> Process<'a, T, F> for MapDF<'a, T, I, F>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
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

    fn mapdf(self, f: F) -> MapDF<'a, T, Self, F>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
    {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}


impl<'a, I, T, F> Process<'a, T, F> for Impute<'a, I, T>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
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

    fn mapdf(self, f: F) -> MapDF<'a, T, Self, F> {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}

impl<'a, T, F> Process<'a, T, F> for MutableDataFrameIterator<'a, T>
    where T: UtahNum,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
    {

        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a, T>
        where Self: Sized + Iterator<Item = (String, ArrayViewMut1<'a, T>)>
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

    fn mapdf(self, f: F) -> MapDF<'a, T, Self, F> {
        let axis = self.axis.clone();
        let other = self.other.clone();
        MapDF::new(self, f, other, axis)
    }
}

impl<'a, T> ToDataFrame<'a, (String, ArrayViewMut1<'a, T>), T> for MutableDataFrameIterator<'a, T>
    where T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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

impl<'a, I, T> ToDataFrame<'a, (String, ArrayViewMut1<'a, T>), T> for Impute<'a, I, T>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum
{
    fn as_df(self) -> Result<DataFrame<T>> {
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



impl<'a, I, T, F> ToDataFrame<'a, (String, ArrayViewMut1<'a, T>), T> for MapDF<'a, T, I, F>
    where I: Iterator<Item = (String, ArrayViewMut1<'a, T>)>,
          T: UtahNum,
          F: Fn(T) -> T
{
    fn as_df(self) -> Result<DataFrame<T>> {
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
