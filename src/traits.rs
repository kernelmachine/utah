use types::*;
use std::iter::{Iterator, Chain};
use aggregate::*;
use transform::*;
use impute::*;
pub trait DFIter<'a> {
    type DFItem;

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              OuterType: From<&'a T>,
              T: 'a;
    fn concat<I>(self, other: I) -> Chain<Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn inner_left_join<I>(self, other: I) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn outer_left_join<I>(self, other: I) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn inner_right_join<I>(self, other: I) -> InnerJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn outer_right_join<I>(self, other: I) -> OuterJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              F: Fn(&InnerType) -> B;
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>;
}
