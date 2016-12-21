
use types::*;
use std::iter::{Iterator, Chain};
use aggregate::*;
use transform::*;
use process::*;
use dataframe::{DataFrame, MutableDataFrame};
use std::hash::Hash;
use std::fmt::Debug;
use error::*;
use join::*;
use std::ops::{Add, Sub, Mul, Div};
use num::traits::One;
use std::collections::BTreeMap;

pub trait Empty<T> {
    fn empty() -> T;
    fn is_empty(&self) -> bool;
}


pub trait MixedDataframeConstructor<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
        T: 'a + Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+One,
          S: 'a + Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>,
          Self : Sized
{
    fn new<U: Clone + Debug + Ord>(data: BTreeMap<U, Row<T>>) -> Self where S: From<U>, U : 'a;
    fn index<U: Clone + Ord>(self, index: &'a [U]) -> Result<Self> where S: From<U>;
    fn columns<U: Clone + Ord>(self, columns: &'a [U]) -> Result<Self> where S: From<U>;
    fn from_array(data: Row<T>, axis: UtahAxis) -> Self;
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, I, T, S>;
    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a, T, S>;
}


pub trait Constructor<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>> + Clone,
            T: 'a + Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>,
          Self : Sized
{
    fn new<U: Clone + Debug>(data: Matrix<U>) -> Self where T: From<U>;
    fn index<U: Clone>(self, index: &'a [U]) -> Result<Self> where S: From<U>;
    fn columns<U: Clone>(self, columns: &'a [U]) -> Result<Self> where S: From<U>;
    fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> Self where T: From<U>;
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, I, T, S>;
    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a,  T, S>;
}


pub trait Operations<'a, I, T, S>
    where I: Iterator<Item = RowView<'a, T>>  + Clone,
        T: 'a + Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
// fn new<U: Clone + Debug>(data: Matrix<U>) -> DataFrame<T, S> where T: From<U>;
// fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> DataFrame<T, S> where T: From<U>;
// fn columns<U: Clone>(self, columns: &'a [U]) -> Result<DataFrame<T, S>> where S: From<U>;
// fn index<U: Clone>(self, index: &'a [U]) -> Result<DataFrame<T, S>> where S: From<U>;
    fn shape(self) -> (usize, usize);
// fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, I, T, S>;
// fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a, I, T, S>;
    fn select<U :?Sized>(&'a self,
                         names: &'a [&'a U],
                         axis: UtahAxis)
                         -> Select<'a, T, S, DataFrameIterator<'a, I, T, S>>
        where S: From<&'a  U >;
    fn remove<U: ?Sized>(&'a self,
                 names: &'a [&'a U],
                 axis: UtahAxis)
                 -> Remove<'a, DataFrameIterator<'a, I, T, S>, T, S>
        where S: From<&'a U>;
    fn append<U :?Sized>(&'a mut self,
                 name: &'a U,
                 data: RowView<'a, T>,
                 axis: UtahAxis)
                 -> Append<'a, DataFrameIterator<'a, I, T, S>, T, S>
        where S: From<&'a U>;
    fn inner_left_join(&'a self,
                       other: &'a DataFrame<T, S>)
                       -> InnerJoin<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn outer_left_join(&'a self,
                       other: &'a DataFrame<T, S>)
                       -> OuterJoin<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn inner_right_join(&'a self,
                        other: &'a DataFrame<T, S>)
                        -> InnerJoin<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn outer_right_join(&'a self,
                        other: &'a DataFrame<T, S>)
                        -> OuterJoin<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn sumdf(&'a mut self, axis: UtahAxis) -> Sum<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn map<F, B>(&'a mut self,
                 f: F,
                 axis: UtahAxis)
                 -> MapDF<'a, T, S, DataFrameIterator<'a, I, T, S>, F, B>
        where F: Fn(&T) -> B,
              for<'r> F: Fn(&InnerType) -> B;
    fn mean(&'a mut self, axis: UtahAxis) -> Mean<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn maxdf(&'a mut self, axis: UtahAxis) -> Max<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn min(&'a mut self, axis: UtahAxis) -> Min<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn stdev(&'a self, axis: UtahAxis) -> Stdev<'a, DataFrameIterator<'a, I, T, S>, T, S>;
    fn impute(&'a mut self,
              strategy: ImputeStrategy,
              axis: UtahAxis)
              -> Impute<'a, MutableDataFrameIterator<'a, T, S>, T, S>;
}

pub trait Aggregate<'a, T, S>
    where T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn sumdf(self) -> Sum<'a, Self, T, S> where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>;

    fn maxdf(self) -> Max<'a, Self, T, S> where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>;

    fn mindf(self) -> Min<'a, Self, T, S> where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>;

    fn mean(self) -> Mean<'a, Self, T, S> where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>;

    fn stdev(self) -> Stdev<'a, Self, T, S> where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>;
}
pub trait Process<'a, T, S>
    where T: Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug + From<String>
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>;
    fn to_mut_df(self) -> MutableDataFrame<'a, T, S>
        where Self: Sized + Iterator<Item = (S, RowViewMut<'a, T>)>;
}

pub trait Transform<'a, T, S>
    where T: Clone + Debug + 'a,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, T, S, Self>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>,
              T: 'a;
    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>,
              T: 'a;
    fn append<U: ?Sized>(self, name: &'a U, data: RowView<'a, T>) -> Append<'a, Self, T, S>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              S: From<&'a U>,
              T: 'a;
    fn concat<I>(self, other: I) -> Chain<Self, I>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)>,
              I: Sized + Iterator<Item = (S, RowView<'a, T>)>;

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, T, S, Self, F, B>
        where Self: Sized + Iterator<Item = (S, RowView<'a, T>)> + Clone,
              F: Fn(&T) -> B;
}



pub trait ToDataFrame<'a, I, T, S>
    where T: Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+ One,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>
{
    fn as_df(self) -> DataFrame<T, S> where Self: Sized + Iterator<Item = I>;
    fn as_matrix(self) -> Matrix<T> where Self : Sized + Iterator<Item = I>;
    fn as_array(self) -> Row<T> where Self : Sized + Iterator<Item = I>;
}
