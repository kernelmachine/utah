use ndarray::{Array, ArrayView, ArrayViewMut, Ix};
use adapters::transform::*;
use adapters::join::*;
use adapters::aggregate::*;
use adapters::process::*;
use std::iter::Chain;
use dataframe::{DataFrameIterator, MutableDataFrameIterator};


#[derive( Clone, Debug, Copy)]
pub enum UtahAxis {
    Row,
    Column,
}

#[derive( Clone, Debug)]
pub enum ImputeStrategy {
    Mean,
}


pub type Column<T> = Array<T, Ix>;
pub type Row<T> = Array<T, Ix>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type MatrixMut<'a, T> = Array<&'a mut T, (Ix, Ix)>;

pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type RowView<'a, T> = ArrayView<'a, T, Ix>;
pub type RowViewMut<'a, T> = ArrayViewMut<'a, T, Ix>;

pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;

pub type DFIter<'a, T, S> = DataFrameIterator<'a, T, S>;
pub type AppendIter<'a, T, S> = Append<'a, DFIter<'a, T, S>, T, S>;
pub type SelectIter<'a, T, S> = Select<'a, DFIter<'a, T, S>, T, S>;
pub type RemoveIter<'a, T, S> = Remove<'a, DFIter<'a, T, S>, T, S>;
pub type InnerJoinIter<'a, T, S> = InnerJoin<'a, DFIter<'a, T, S>, T, S>;
pub type OuterJoinIter<'a, T, S> = OuterJoin<'a, DFIter<'a, T, S>, T, S>;
pub type ConcatIter<'a, T, S> = Concat<'a, Chain<DFIter<'a, T, S>, DFIter<'a, T, S>>, T, S>;
pub type SumIter<'a, T, S> = Sum<'a, DFIter<'a, T, S>, T, S>;
pub type MaxIter<'a, T, S> = Max<'a, DFIter<'a, T, S>, T, S>;
pub type MinIter<'a, T, S> = Min<'a, DFIter<'a, T, S>, T, S>;
pub type StdevIter<'a, T, S> = Stdev<'a, DFIter<'a, T, S>, T, S>;
pub type MeanIter<'a, T, S> = Mean<'a, DFIter<'a, T, S>, T, S>;
pub type MapDFIter<'a, T, S, F, B> = MapDF<'a, T, S, DFIter<'a, T, S>, F, B>;
pub type ImputeIter<'a, T, S> = Impute<'a, MutableDataFrameIterator<'a, T, S>, T, S>;
