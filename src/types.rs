use chrono::*;
use ndarray::{Array, ArrayView, Ix};
use num::One;
use std::ops::{Mul, Add, Sub};

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
    Empty,
}


pub type Column<T> = Array<T, Ix>;
pub type Row<T> = Array<T, Ix>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type RowView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;



impl Mul for InnerType {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match self {
            InnerType::Float(x) => {
                match rhs {
                    InnerType::Float(y) => InnerType::Float(x * y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int32(x) => {
                match rhs {
                    InnerType::Int32(y) => InnerType::Int32(x * y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int64(x) => {
                match rhs {
                    InnerType::Int64(y) => InnerType::Int64(x * y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Str(_) => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
            InnerType::Empty => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
        }
    }
}

impl Add for InnerType {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match self {
            InnerType::Float(x) => {
                match rhs {
                    InnerType::Float(y) => InnerType::Float(x + y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int32(x) => {
                match rhs {
                    InnerType::Int32(y) => InnerType::Int32(x + y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int64(x) => {
                match rhs {
                    InnerType::Int64(y) => InnerType::Int64(x + y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Str(_) => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
            InnerType::Empty => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
        }
    }
}

impl Sub for InnerType {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        match self {
            InnerType::Float(x) => {
                match rhs {
                    InnerType::Float(y) => InnerType::Float(x - y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int32(x) => {
                match rhs {
                    InnerType::Int32(y) => InnerType::Int32(x - y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int64(x) => {
                match rhs {
                    InnerType::Int64(y) => InnerType::Int64(x - y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Str(_) => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
            InnerType::Empty => {
                match rhs {
                    _ => InnerType::Empty,
                }
            }
        }
    }
}

impl One for InnerType {
    fn one() -> Self {
        InnerType::Float(1.0)
    }
}
