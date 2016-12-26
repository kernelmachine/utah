use util::error::*;
use util::types::*;
use std::iter::Iterator;
use ndarray::{AxisIter, AxisIterMut};
use util::types::UtahAxis;
use util::traits::*;
use std::slice::Iter;


/// A read-only dataframe.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<T, S>
    where T: Num,
          S: Identifier
{
    pub columns: Vec<S>,
    pub data: Matrix<T>,
    pub index: Vec<S>,
}

/// A read-write dataframe
#[derive(Debug, PartialEq)]
pub struct MutableDataFrame<'a, T: 'a, S>
    where T: Num,
          S: Identifier + Clone
{
    pub columns: Vec<S>,
    pub data: MatrixMut<'a, T>,
    pub index: Vec<S>,
}


/// The read-only dataframe iterator
#[derive(Clone)]
pub struct DataFrameIterator<'a, T: 'a, S: 'a>
    where T: Num,
          S: Identifier
{
    pub names: Iter<'a, S>,
    pub data: AxisIter<'a, T, usize>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}




impl<'a, T, S> Iterator for DataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier
{
    type Item = (S, RowView<'a, T>);
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
pub struct MutableDataFrameIterator<'a, T, S>
    where T: Num + 'a,
          S: Identifier + 'a
{
    pub names: Iter<'a, S>,
    pub data: AxisIterMut<'a, T, usize>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, T, S> Iterator for MutableDataFrameIterator<'a, T, S>
    where T: Num,
          S: Identifier
{
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


impl<'a, T, S> MutableDataFrame<'a, T, S>
    where T: 'a + Num,
          S: Identifier
{
    /// Dereference a mutable dataframe as an owned dataframe.
    pub fn to_df(self) -> Result<DataFrame<T, S>> {
        let d = self.data.map(|x| ((*x).clone()));
        let df = DataFrame::new(d).columns(&self.columns[..])?.index(&self.index[..])?;
        Ok(df)

    }
}
