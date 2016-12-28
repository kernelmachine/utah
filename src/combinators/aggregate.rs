//! Utah aggregation combinators.


use util::types::*;
use util::traits::*;
use dataframe::*;
use ndarray::{Array, ArrayView1};
use util::error::*;

#[derive(Clone, Debug)]
pub struct Sum<'a, I: 'a, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)> + 'a,
          T: Num
{
    data: I,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, I, T> Sum<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    pub fn new(df: I, other: Vec<String>, axis: UtahAxis) -> Sum<'a, I, T> {

        Sum {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T> Iterator for Sum<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => return Some(dat.scalar_sum()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mean<'a, I: 'a, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    data: I,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, I, T> Mean<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + 'a
{
    pub fn new(df: I, other: Vec<String>, axis: UtahAxis) -> Mean<'a, I, T> {

        Mean {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T> Iterator for Mean<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + 'a
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            Some((_, dat)) => {
                let size = dat.fold(T::zero(), |acc, _| acc + T::one());
                let mean = dat.scalar_sum() / size;
                Some(mean)
            }
            None => return None,

        }
    }
}


#[derive(Clone)]
pub struct Max<'a, I: 'a, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    data: I,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, I, T> Max<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + 'a
{
    pub fn new(df: I, other: Vec<String>, axis: UtahAxis) -> Max<'a, I, T> {

        Max {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T> Iterator for Max<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + Ord + 'a
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().max().map(|x| x.clone()),
        }



    }
}


#[derive(Clone, Debug)]
pub struct Min<'a, I: 'a, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    data: I,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, I, T> Min<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + 'a
{
    pub fn new(df: I, other: Vec<String>, axis: UtahAxis) -> Min<'a, I, T> {

        Min {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T> Iterator for Min<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + Ord
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().min().map(|x| x.clone()),
        }



    }
}

#[derive(Clone)]
pub struct Stdev<'a, I: 'a, T: 'a>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    data: I,
    other: Vec<String>,
    axis: UtahAxis,
}

impl<'a, I, T> Stdev<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + 'a
{
    pub fn new(df: I, other: Vec<String>, axis: UtahAxis) -> Stdev<'a, I, T> {

        Stdev {
            data: df,
            other: other,
            axis: axis,
        }
    }
}


impl<'a, I, T> ToDataFrame<'a, T, T> for Mean<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    fn as_df(self) -> Result<DataFrame<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec!["0"];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}



impl<'a, I, T> ToDataFrame<'a, T, T> for Max<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + Ord
{
    fn as_df(self) -> Result<DataFrame<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec!["0"];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }

        }

    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }
    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}


impl<'a, I, T> ToDataFrame<'a, T, T> for Min<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num + Ord
{
    fn as_df(self) -> Result<DataFrame<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec!["0"];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}


impl<'a, I, T> ToDataFrame<'a, T, T> for Sum<'a, I, T>
    where I: Iterator<Item = (String, ArrayView1<'a, T>)>,
          T: Num
{
    fn as_df(self) -> Result<DataFrame<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };

        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec!["0"];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (other.len(), 1),
            UtahAxis::Column => (1, other.len()),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}
