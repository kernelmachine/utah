use chrono::*;
use ndarray::{Array, ArrayView, Ix};

#[derive(Hash, Eq ,PartialOrd, PartialEq, Ord , Clone, Debug)]
pub enum ColumnType {
    Str(String),
    Date(DateTime<UTC>),
    Int(i64),
}

#[derive(Hash, PartialOrd, PartialEq, Eq , Ord , Clone,  Debug)]
pub enum IndexType {
    Str(String),
    Date(DateTime<UTC>),
    Int(i64),
}

#[derive(PartialOrd, PartialEq,  Clone, Debug, Copy)]
pub enum InnerType {
    Float(f64),
    Int(i64),
}

#[derive(PartialOrd, PartialEq,  Clone, Debug)]
pub enum SlowerInnerType {
    Float(f64),
    Int(i64),
    Str(String),
}

pub type Column<T> = Array<T, Ix>;
pub type Row<T> = Array<T, Ix>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type RowView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;
