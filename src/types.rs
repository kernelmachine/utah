use chrono::*;
use ndarray::{Array, ArrayView, Ix};


#[derive(Hash, PartialOrd, PartialEq, Eq , Ord , Clone,  Debug)]
pub enum OuterType {
    Str(String),
    Date(DateTime<UTC>),
    Int64(i64),
    Int32(i32),
}

#[derive(PartialOrd, PartialEq,  Clone, Debug)]
pub enum InnerType {
    Float(f64),
    Int64(i64),
    Int32(i32),
    Str(String),
}


pub type Column<T> = Array<T, Ix>;
pub type Row<T> = Array<T, Ix>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type RowView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;
