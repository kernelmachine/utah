use ndarray::{Array2, Array1, ArrayView1, ArrayView2};
use combinators::transform::*;
use combinators::interact::*;
use combinators::aggregate::*;
use combinators::process::*;
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


pub type Column<T> = Array1<T>;
pub type Row<T> = Array1<T>;
pub type RowMut<'a, T> = Array1<&'a mut T>;

pub type Matrix<T> = Array2<T>;
pub type MatrixMut<'a, T> = Array2<&'a mut T>;

pub type ColumnView<'a, T> = ArrayView1<'a, T>;

pub type MatrixView<'a, T> = ArrayView2<'a, T>;

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
pub type MapDFIter<'a, T, S, F> = MapDF<'a, T, S, MutableDataFrameIterator<'a, T, S>, F>;
pub type ImputeIter<'a, T, S> = Impute<'a, MutableDataFrameIterator<'a, T, S>, T, S>;
