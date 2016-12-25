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
    /// Create a new dataframe. The only required argument is data to populate the dataframe. The data's elements can be any of `InnerType`.
    /// By default, the columns and index of the dataframe are `["1", "2", "3"..."N"]`, where *N* is
    /// the number of columns (or rows) in the data.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a);
    /// ```
    ///
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[InnerType::Float(2.0), InnerType::Str("ak".into())],
    ///                [InnerType::Int32(6), InnerType::Int64(10)]]);
    /// let df = DataFrame::new(a);
    /// ```
    pub fn to_df(self) -> Result<DataFrame<T, S>> {
        let d = self.data.map(|x| ((*x).clone()));
        let df = DataFrame::new(d).columns(&self.columns[..])?.index(&self.index[..])?;
        Ok(df)

    }
}
