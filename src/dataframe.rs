//! Utah dataframe

use util::error::*;
use util::types::*;
use std::iter::Iterator;
use ndarray::{AxisIter, AxisIterMut};
use util::traits::*;
use std::slice::Iter;
use ndarray::{ArrayView1, ArrayViewMut1, Dim, Ix};

/// A read-only dataframe.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<T>
    where T: UtahNum
{
    pub columns: Vec<String>,
    pub data: Matrix<T>,
    pub index: Vec<String>,
}

/// A read-write dataframe
#[derive(Debug, PartialEq)]
pub struct MutableDataFrame<'a, T: 'a>
    where T: UtahNum
{
    pub columns: Vec<String>,
    pub data: MatrixMut<'a, T>,
    pub index: Vec<String>,
}


/// The read-only dataframe iterator
#[derive(Clone)]
pub struct DataFrameIterator<'a, T: 'a>
    where T: UtahNum
{
    pub names: Iter<'a, String>,
    pub data: AxisIter<'a, T, Dim<[Ix; 1]>>,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}




impl<'a, T> Iterator for DataFrameIterator<'a, T>
    where T: UtahNum
{
    type Item = (String, ArrayView1<'a, T>);
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

/// The read-write dataframe iterator
pub struct MutableDataFrameIterator<'a, T>
    where T: UtahNum + 'a
{
    pub names: Iter<'a, String>,
    pub data: AxisIterMut<'a, T, Dim<[Ix; 1]>>,
    pub other: Vec<String>,
    pub axis: UtahAxis,
}


impl<'a, T> Iterator for MutableDataFrameIterator<'a, T>
    where T: UtahNum
{
    type Item = (String, ArrayViewMut1<'a, T>);


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


impl<'a, T> MutableDataFrame<'a, T>
    where T: 'a + UtahNum
{
    /// Dereference a mutable dataframe as an owned dataframe.
    pub fn to_df(self) -> Result<DataFrame<T>> {
        let d = self.data.map(|x| ((*x).clone()));
        let df = DataFrame::new(d).columns(&self.columns[..])?.index(&self.index[..])?;
        Ok(df)

    }
}
