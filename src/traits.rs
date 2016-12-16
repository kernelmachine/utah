
use types::*;
use std::iter::{Iterator, Chain};
use aggregate::*;
use transform::*;
use process::*;
use dataframe::{DataFrame, MutableDataFrame};


pub trait ToDataFrame<'a> {
    fn to_df(self) -> DataFrame where Self: Sized + Iterator<Item = InnerType>;
}
pub trait Aggregate<'a> {
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn maxdf(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mindf(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
}
pub trait Process<'a> {
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>;
    fn to_mut_df(self) -> MutableDataFrame<'a>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>;
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>;
}
pub trait Transform<'a> {
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn concat<I>(self, other: I) -> Chain<Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              F: Fn(&InnerType) -> B;
}

pub trait Join<'a> {
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>, RowView<'a, InnerType>)>;
}
