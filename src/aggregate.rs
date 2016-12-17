
use types::*;
use traits::*;
use dataframe::*;
use ndarray::Array;
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone)]
pub struct Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Sum<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)>
    {

        Sum {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => {
                return Some((0..dat.len()).fold(dat.get(0).unwrap().to_owned(),
                                                |x, y| x + dat.get(y).unwrap().to_owned()))
            }
        }
    }
}

#[derive(Clone)]
pub struct Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Mean<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)>
    {

        Mean {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned() / size);
                Some(sum)
                // match dat.uget(0) {
                //     &InnerType::Float(_) => return Some(sum / InnerType::Float(size as f64)),
                //     &InnerType::Int32(_) => return Some(sum / InnerType::Int32(size as i32)),
                //     &InnerType::Int64(_) => return Some(sum / InnerType::Int64(size as i64)),
                //     _ => return Some(InnerType::Empty),
                // }

            },
        }
    }
}


#[derive(Clone)]
pub struct Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Max<'a, I, T, S> {

        Max {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a + Ord,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().max().map(|x| x.to_owned()),
        }



    }
}


#[derive(Clone)]
pub struct Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Min<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)>,
              T: Clone + Debug + 'a,
              S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
    {

        Min {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().min().map(|x| x.to_owned()),
        }



    }
}

#[derive(Clone)]
pub struct Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Stdev<'a, I, T, S>
        where I: Iterator<Item = (S, RowView<'a, T>)>
    {

        Stdev {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let mean = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned() / size);

                // let mean = match dat.uget(0) {
                //     &InnerType::Float(_) => sum / InnerType::Float(size as f64),
                //     &InnerType::Int32(_) => sum / InnerType::Int32(size as i32),
                //     &InnerType::Int64(_) => sum / InnerType::Int64(size as i64),
                //     _ => InnerType::Empty,
                // };

                let stdev = (0..size).fold(dat.uget(0).to_owned(), |x, y| {
                    x +
                    (dat.uget(y).to_owned() - mean.to_owned()) *
                    (dat.uget(y).to_owned() - mean.to_owned())
                });


                Some(stdev)


            },
        }



    }
}


impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + Clone,
          T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn to_df(self) -> DataFrame<T, S> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType, InnerType, OuterType> for Mean<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}



impl<'a, I> ToDataFrame<'a, InnerType, InnerType, OuterType> for Max<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType, InnerType, OuterType> for Min<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType, InnerType, OuterType> for Sum<'a, I, InnerType, OuterType>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame<InnerType, OuterType> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}
