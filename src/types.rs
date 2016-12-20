use chrono::*;
use ndarray::{Array, ArrayView, ArrayViewMut, Ix};
use std::ops::{Mul, Add, Sub, Div};
use std::cmp::Ordering;
use traits::Empty;
use std::f64::NAN;
use std::default::Default;
use num::traits::One;

#[derive(Hash, PartialOrd, PartialEq, Eq , Ord , Clone,  Debug)]
pub enum OuterType {
    Str(String),
    Date(DateTime<UTC>),
    Int64(i64),
    Int32(i32),
    USize(usize),
}

#[derive(Clone, Debug)]
pub enum InnerType {
    Float(f64),
    Int64(i64),
    Int32(i32),
    Str(String),
    Empty,
}

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






impl AsMut<InnerType> for InnerType {
    fn as_mut(&mut self) -> &mut InnerType {
        &mut (*self)
    }
}

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
                    InnerType::Float(y) => InnerType::Float(x as f64 * y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int64(x) => {
                match rhs {
                    InnerType::Int64(y) => InnerType::Int64(x * y),
                    InnerType::Float(y) => InnerType::Float(x as f64 * y),
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
                    InnerType::Float(y) => InnerType::Float(y),
                    InnerType::Int32(y) => InnerType::Int32(y),
                    InnerType::Int64(y) => InnerType::Int64(y),
                    _ => InnerType::Empty,
                }
            }
        }
    }
}


impl Eq for InnerType {}


impl Ord for InnerType {
    fn cmp(&self, other: &InnerType) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for InnerType {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self {
            &InnerType::Float(x) => {
                match rhs {
                    &InnerType::Float(y) => Some((x as i32).cmp(&(y as i32))),
                    &InnerType::Empty => Some((x as i32).cmp(&(x as i32 - 1))),
                    _ => panic!(),
                }
            }
            &InnerType::Int32(x) => {
                match rhs {
                    &InnerType::Int32(y) => Some(x.cmp(&y)),
                    &InnerType::Empty => Some(x.cmp(&(x - 1))),
                    _ => panic!(),
                }
            }
            &InnerType::Int64(x) => {
                match rhs {
                    &InnerType::Int64(y) => Some(x.cmp(&y)),
                    &InnerType::Empty => Some(x.cmp(&(x - 1))),
                    _ => panic!(),
                }
            }
            &InnerType::Str(ref x) => {
                match rhs {
                    &InnerType::Str(ref y) => Some(x.cmp(&y)),
                    _ => panic!(),
                }
            }
            &InnerType::Empty => {
                match rhs {
                    &InnerType::Float(y) => Some((y as i32).cmp(&(y as i32 - 1))),
                    &InnerType::Int64(y) => Some(y.cmp(&(y - 1))),
                    &InnerType::Int32(y) => Some(y.cmp(&(y - 1))),
                    _ => panic!(),
                }
            }
        }
    }
}


impl PartialEq for InnerType {
    fn eq(&self, rhs: &Self) -> bool {
        match self {
            &InnerType::Float(x) => {
                match rhs {
                    &InnerType::Float(y) => x == y,
                    _ => false,
                }
            }
            &InnerType::Int32(x) => {
                match rhs {
                    &InnerType::Int32(y) => x == y,
                    _ => false,
                }
            }
            &InnerType::Int64(ref x) => {
                match rhs {
                    &InnerType::Int64(y) => x.to_owned() == y,
                    _ => false,
                }
            }
            &InnerType::Str(ref x) => {
                match rhs {
                    &InnerType::Str(ref y) => x.to_owned() == y.to_owned(),
                    _ => false,
                }
            }
            &InnerType::Empty => {
                match rhs {
                    &InnerType::Empty => true,
                    _ => false,
                }
            }
        }
    }
}


impl Div for InnerType {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        match self {
            InnerType::Float(x) => {
                match rhs {
                    InnerType::Float(y) => InnerType::Float(x / y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int32(x) => {
                match rhs {
                    InnerType::Int32(y) => InnerType::Int32(x / y),
                    _ => InnerType::Empty,
                }
            }
            InnerType::Int64(x) => {
                match rhs {
                    InnerType::Int64(y) => InnerType::Int64(x / y),
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
                    InnerType::Float(y) => InnerType::Float(y),
                    InnerType::Int32(y) => InnerType::Int32(y),
                    InnerType::Int64(y) => InnerType::Int64(y),
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
                    InnerType::Float(y) => InnerType::Float(y),
                    InnerType::Int32(y) => InnerType::Int32(y),
                    InnerType::Int64(y) => InnerType::Int64(y),
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
                    InnerType::Float(y) => InnerType::Float(y),
                    InnerType::Int32(y) => InnerType::Int32(y),
                    InnerType::Int64(y) => InnerType::Int64(y),
                    _ => InnerType::Empty,
                }
            }
        }
    }
}
// impl One for InnerType {
//     fn one() -> InnerType {
//         InnerType::Float(1.0)
//     }
// }
//
// impl Zero for InnerType {
//     fn zero() -> InnerType {
//         InnerType::Float(0.0)
//     }
//     fn is_zero(&self) -> bool {
//         *self == InnerType::Float(0.0)
//     }
// }

impl Empty<InnerType> for InnerType {
    fn empty() -> InnerType {
        InnerType::Empty
    }
    fn is_empty(&self) -> bool {
        match *self {
            InnerType::Float(x) => x == NAN,
            InnerType::Empty => true,
            _ => false,
        }
    }
}

impl Empty<f64> for f64 {
    fn empty() -> f64 {
        NAN
    }
    fn is_empty(&self) -> bool {
        self.is_nan()
    }
}

impl Empty<i32> for i32 {
    fn empty() -> i32 {
        0
    }
    fn is_empty(&self) -> bool {
        *self == 0
    }
}

impl One for InnerType {
    fn one() -> InnerType {
        InnerType::Float(1.0)
    }
}

impl Default for OuterType {
    fn default() -> OuterType {
        OuterType::Int32(1)
    }
}

impl Default for InnerType {
    fn default() -> InnerType {
        InnerType::Int32(1)
    }
}
