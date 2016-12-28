
use util::types::*;
use std::iter::Iterator;
use combinators::aggregate::*;
use combinators::transform::*;
use combinators::process::*;
use dataframe::{DataFrame, DataFrameMut, DataFrameIterator, DataFrameMutIterator};
use std::fmt::Debug;
use util::error::*;
use std::ops::{Add, Sub, Mul, Div};
use num::traits::{One, Zero};
use ndarray::ArrayView1;

pub trait UtahNum
    : Add<Output = Self> +
      Div<Output = Self> +
      Sub<Output = Self> +
      Mul<Output = Self> +
      Empty<Self> +
      One +
      Zero +
      Clone +
      Debug +
      PartialEq +
      Default
    {}

impl<T> UtahNum for T
    where T: Add<Output = T> +
             Div<Output = T> +
             Sub<Output = T> +
             Mul<Output = T> +
             Empty<T> +
             One +
             Zero +
             Clone +
             Debug +
             PartialEq +
             Default
{
}


pub trait Empty<T> {
    fn empty() -> T;
    fn is_empty(&self) -> bool;
}

pub trait Constructor<'a, T>
    where T: 'a + UtahNum,
          Self: Sized
{
    fn new<U: Clone + Debug>(data: Matrix<U>) -> DataFrame<T> where T: From<U>;
    fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> DataFrame<T> where T: From<U>;
    fn index<U: Clone>(self, index: &'a [U]) -> Result<Self> where String: From<U>;
    fn columns<U: Clone>(self, columns: &'a [U]) -> Result<Self> where String: From<U>;
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, T>;
    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> DataFrameMutIterator<'a, T>;
}


pub trait Operations<'a, T>
    where T: 'a + UtahNum
{
    fn shape(self) -> (usize, usize);
    fn select<U: ?Sized>(&'a self, names: &'a [&'a U], axis: UtahAxis) -> SelectIter<'a, T>
        where String: From<&'a U>;
    fn remove<U: ?Sized>(&'a self, names: &'a [&'a U], axis: UtahAxis) -> RemoveIter<'a, T>
        where String: From<&'a U>;
    fn append<U: ?Sized>(&'a mut self,
                         name: &'a U,
                         data: ArrayView1<'a, T>,
                         axis: UtahAxis)
                         -> AppendIter<'a, T>
        where String: From<&'a U>;
    fn inner_left_join(&'a self, other: &'a DataFrame<T>) -> InnerJoinIter<'a, T>;
    fn outer_left_join(&'a self, other: &'a DataFrame<T>) -> OuterJoinIter<'a, T>;
    fn inner_right_join(&'a self, other: &'a DataFrame<T>) -> InnerJoinIter<'a, T>;
    fn outer_right_join(&'a self, other: &'a DataFrame<T>) -> OuterJoinIter<'a, T>;
    fn concat(&'a self, other: &'a DataFrame<T>, axis: UtahAxis) -> ConcatIter<'a, T>;
    fn sumdf(&'a mut self, axis: UtahAxis) -> SumIter<'a, T>;
    fn mean(&'a mut self, axis: UtahAxis) -> MeanIter<'a, T>;
    fn maxdf(&'a mut self, axis: UtahAxis) -> MaxIter<'a, T>;
    fn mindf(&'a mut self, axis: UtahAxis) -> MinIter<'a, T>;
    fn mapdf<F>(&'a mut self, f: F, axis: UtahAxis) -> MapDFIter<'a, T, F>
        where F: Fn(T) -> T,
              for<'r> F: Fn(T) -> T;
    fn impute(&'a mut self, strategy: ImputeStrategy, axis: UtahAxis) -> ImputeIter<'a, T>;
}

pub trait Aggregate<'a, T>
    where T: UtahNum
{
    fn sumdf(self) -> Sum<'a, Self, T> where Self: Sized + Iterator<Item = Window<'a, T>>;

    fn maxdf(self) -> Max<'a, Self, T> where Self: Sized + Iterator<Item = Window<'a, T>>;

    fn mindf(self) -> Min<'a, Self, T> where Self: Sized + Iterator<Item = Window<'a, T>>;

    fn mean(self) -> Mean<'a, Self, T> where Self: Sized + Iterator<Item = Window<'a, T>>;
}

pub trait Process<'a, T, F>
    where T: UtahNum,
          F: Fn(T) -> T
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self, T>
        where Self: Sized + Iterator<Item = WindowMut<'a, T>>;
    fn to_mut_df(self) -> DataFrameMut<'a, T> where Self: Sized + Iterator<Item = WindowMut<'a, T>>;
    fn mapdf(self, f: F) -> MapDF<'a, T, Self, F>
        where Self: Sized + Iterator<Item = WindowMut<'a, T>> + Clone;
}

pub trait Transform<'a, T>
    where T: UtahNum + 'a
{
    fn select<U: ?Sized>(self, names: &'a [&'a U]) -> Select<'a, Self, T>
        where Self: Sized + Iterator<Item = Window<'a, T>> + Clone,
              String: From<&'a U>,
              T: 'a;
    fn remove<U: ?Sized>(self, names: &'a [&'a U]) -> Remove<'a, Self, T>
        where Self: Sized + Iterator<Item = Window<'a, T>> + Clone,
              String: From<&'a U>,
              T: 'a;
    fn append<U: ?Sized>(self, name: &'a U, data: ArrayView1<'a, T>) -> Append<'a, Self, T>
        where Self: Sized + Iterator<Item = Window<'a, T>> + Clone,
              String: From<&'a U>,
              T: 'a;
}



pub trait ToDataFrame<'a, I, T>
    where T: UtahNum + 'a
{
    fn as_df(self) -> Result<DataFrame<T>> where Self: Sized + Iterator<Item = I>;
    fn as_matrix(self) -> Result<Matrix<T>> where Self: Sized + Iterator<Item = I>;
    fn as_array(self) -> Result<Row<T>> where Self: Sized + Iterator<Item = I>;
}
