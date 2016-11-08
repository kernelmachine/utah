#[allow(dead_code)]

use ndarray::{Array, Ix, Axis, ArrayView, stack};
use std::collections::HashMap;
use std::mem;


pub type Column = Array<Data, Ix>;
pub type Row = Vec<Data>;

pub type Matrix = Array<Data, (Ix, Ix)>;
pub type ColumnView<'a> = ArrayView<'a, Data, Ix>;
pub type MatrixView<'a> = ArrayView<'a, Data, (Ix, Ix)>;
pub type ArrayColumn = Array<Data, Ix>;
pub type ArrayRow = Array<Data, Ix>;

#[derive(PartialEq, PartialOrd, Clone, Debug, Copy)]
pub enum Data {
    Int(i64),
    Float(f64), // String,
}

// impl Data {
//     fn from_int(i: i64) -> Data {
//         Data::Int(i)
//     }
//     fn from_float(f: f64) -> Data {
//         Data::Float(f)
//     }
//     fn from_string(s: String) -> Data {
//          Data::Str(s)
//      }
// }

macro_rules! dataframe {
    ($name : ident, {$($field : ident : $value : expr),*}) => {
        #[derive(Debug, PartialEq, PartialOrd)]
        struct $name {$($field: Vec<Data>),*};
        impl Default for $name {
            fn default() -> $name {
                $name {$($field: $value),*}
            }
        }
    };
}


// macro_rules! dataframe {
//     ($name : ident,($($field: expr),*), ($($value: expr),*)) => {
//         #[derive(Debug, PartialEq)]
//         struct $name { $($field: ndarray::Array<f64, Ix>),* };
//         impl Default for $name {
//             fn default() -> $name {
//                 $name {
//                 $($field : $value),*
//                 }
//             }
//         }
//
//
//
//     };
// }
//
#[derive(Debug, Clone, PartialEq)]
pub struct InnerMatrix {
    inner_matrix: Matrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataMap {
    data_map: HashMap<String, usize>,
    inner_matrix: InnerMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    data: DataMap,
}


impl DataFrame {
    pub fn inner_matrix(&self) -> &InnerMatrix {
        &self.data.inner_matrix
    }
    pub fn datamap(&self) -> &DataMap {
        &self.data
    }
}

impl DataFrame {
    pub fn new(data: Matrix, names: Vec<String>) -> Result<DataFrame, &'static str> {
        // let vs: Vec<ColumnView> = datamap.values().map(|x| x.view()).collect();
        //
        // let data = stack(Axis(1), &vs[..]).unwrap().into_shape((nrows, ncols)).unwrap();
        if names.len() != data.shape()[1] {
            return Err("shape mismatch!");
        }


        let datamap: HashMap<String, usize> = names.iter()
            .enumerate()
            .map(|(x, y)| (y.clone(), x))
            .collect();
        let mat = InnerMatrix { inner_matrix: data };

        let dm = DataMap {
            data_map: datamap,
            inner_matrix: mat,
        };

        Ok(DataFrame { data: dm })
    }



    pub fn inner_join(self, other: DataFrame, on: &str) -> Result<DataFrame, &'static str> {
        let h_idx: &usize = self.data
            .data_map
            .get(on)
            .unwrap();
        let h: HashMap<Value, usize> = self.data
            .inner_matrix
            .inner_matrix
            .column(*h_idx)
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y).unwrap(), x))
            .collect();
        let j_idx: &usize = other.data
            .data_map
            .get(on)
            .unwrap();
        let j: HashMap<Value, usize> = other.data
            .inner_matrix
            .inner_matrix
            .column(*j_idx)
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y).unwrap(), x))
            .collect();

        let mut idxs = vec![];
        let mut vs = vec![];

        for row_key in h.keys() {
            match j.get(row_key) {
                None => continue,
                Some(v) => {
                    match h.get(row_key) {
                        None => return Err("omg!"),
                        Some(idx) => {
                            idxs.push(*idx);
                            vs.push(*v);
                        }
                    }
                }
            };
        }

        if idxs.len() == 0 && vs.len() == 0 {
            return Err("no common values");
        }
        let new_matrix: Matrix = {
            let mut rng: Vec<usize> = (0..other.data.data_map.len()).collect();
            rng.retain(|x| x != other.data.data_map.get(on).unwrap());
            let m1 = other.data.inner_matrix.inner_matrix.select(Axis(1), &rng[..]);
            let mat1 = self.data.inner_matrix.inner_matrix.select(Axis(0), &idxs[..]);
            let mat2 = m1.select(Axis(0), &vs[..]);
            stack(Axis(1), &[mat1.view(), mat2.view()]).unwrap()
        };

        let mut n1: Vec<String> = self.data.data_map.keys().map(|x| x.clone()).collect();

        let mut n2: Vec<String> = other.data.data_map.keys().map(|x| x.clone()).collect();
        n2.retain(|x| x != on);
        n1.extend_from_slice(&n2[..]);
        n1.dedup();

        DataFrame::new(new_matrix, n1)
    }
}


#[derive(Hash, Eq, Debug, Clone,PartialEq)]
struct Value((u64, i16, i8));

impl Value {
    fn new(val: Data) -> Result<Value, &'static str> {
        match val {
            Data::Float(f) => Ok(Value(integer_decode(f))),
            _ => Err("expected Float"),
        }

    }
}

pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}





impl DataFrame {
    // pub fn new(data: Matrix, names: Vec<String>) -> Result<DataFrame<'a>, &'static str> {
    //     if data.cols() != names.len() {
    //         return Err("mismatched dimensions");
    //     }
    //     let mut df: DataFrame = DataFrame::default();
    //     df.inner_matrix = data;
    //
    //     df.data_map = names.iter()
    //         .zip(df.inner_matrix.axis_iter(Axis(1)))
    //         .map(|(x, y)| (x.to_string(), y))
    //         .collect();
    //
    //
    //     Ok(df.clone())
    // }
    //
}





// // The following implementation requires a copy, which we don't want!
// // Alternative solution to inserting is to mutate the original matrix, but this also requires a
// // copy!
// // fn insert(&mut self, name: &'a str, value: Column) {
// //     match self.data.entry(name) {
// //         Entry::Occupied(mut o) => {
// //             o.get_mut();
// //         }
// //         Entry::Vacant(v) => {
// //             v.insert(value.view());
// //         }
// //     };
// // }
//
//
//
//
// // parallelized join
// // parallelized concatenation
// // parallelized frequency counts
// // index dataframe?
// // sample rows
// // find/select
// // sort
// // statistics (mean, median, stdev)
// // print
//
// // statistics (mean, median, stdev)
// // print
