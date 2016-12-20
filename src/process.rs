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
use num::traits::{One, Zero};

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
    T: Clone + Debug + 'a  + PartialEq + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> +One + Zero,
  S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    type Item = (S, RowViewMut<'a, T>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((val, mut dat)) => {
                match self.strategy {
                    ImputeStrategy::Mean =>{

                        let size = dat.fold(T::one(), |acc, _| acc + T::one());
                        let first_element = unsafe {
                         dat.uget(0).to_owned()
                        };
                        let sum = dat.iter().filter(|&x| !x.is_empty()).fold(first_element, |acc, y| acc + y.clone());

                        let mean = sum/size;

                        dat.mapv_inplace(|x| {
                            if x.is_empty() {
                                mean.to_owned()
                            }
                            else{
                                x.to_owned()
                            }

                        });
                        Some((val, dat))
                    }

// ImputeStrategy::Mode => {
//     let max = dat.iter().max().map(|x| x.to_owned()).unwrap();
//     dat.mapv_inplace(|x| {
//         if x == emp {
//             max.to_owned()
//         }
//         else{
//             x.to_owned()
//         }
//
//     });
//     return Some((val, dat));
// }
                }
            }
        }
    }
}


impl<'a, I,T,S> Process<'a, T, S> for Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
     T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> + One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a + From<String>
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
}

impl<'a, T, S> Process<'a, T, S> for MutableDataFrameIterator<'a, T, S>
where T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> + One,
      S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a + From<String>
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
}

impl<'a, T, S> ToDataFrame<'a, (S, RowViewMut<'a, T>), T, S>
    for MutableDataFrameIterator<'a, T, S>
    where T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> + One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a+ From<String>
          {
    fn as_df(self) -> DataFrame<T, S> {
        self.to_mut_df().to_df()
    }
    fn as_matrix(self) -> Matrix<T> {
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

        Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone()))


    }

    fn as_array(self) -> Row<T> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j);
        }
        Array::from_vec(c).map(|x| ((*x).clone()))
    }

}

impl<'a, I,T,S> ToDataFrame<'a, (S, RowViewMut<'a, T>), T, S> for Impute<'a, I, T, S>
    where I: Iterator<Item = (S, RowViewMut<'a, T>)>,
     T: Clone + Debug  + PartialEq + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> + One + Zero,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug +'a + From<String>
{
    fn as_df(self) -> DataFrame<T, S> {
        self.to_mut_df().to_df()
    }
    fn as_matrix(self) -> Matrix<T> {
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

        Array::from_shape_vec((nrows, ncols), c).unwrap().map(|x| ((*x).clone()))
    }

    fn as_array(self) -> Row<T> {
        let mut c = Vec::new();
        for (_, j) in self {
            c.extend(j);
        }
        Array::from_vec(c).map(|x| ((*x).clone()))
    }
}
