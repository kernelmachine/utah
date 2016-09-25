
use ndarray::{Array, Ix, Axis, ArrayView, stack, ShapeError};
use std::collections::HashMap;
use std::collections::hash_map::Entry;


pub type Column = Array<f64, Ix>;
pub type Matrix = Array<f64, (Ix, Ix)>;
pub type ColumnView<'a> = ArrayView<'a, f64, Ix>;
pub type MatrixView<'a> = ArrayView<'a, f64, (Ix, Ix)>;

macro_rules! dataframe {
    ($name : ident,($($field: expr),*), ($($value: expr),*)) => {
        #[derive(Debug, PartialEq)]
        struct $name { $($field: ndarray::Array<f64, Ix>),* };
        impl Default for $name {
            fn default() -> $name {
                $name {
                $($field : $value),*
                }
            }
        }



    };
}

use std::mem;

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

#[derive(Debug, PartialEq)]
pub struct DataFrame<'b> {
    data: &'b Matrix,
    names: &'b Vec<&'b str>,
    data_map: HashMap<&'b str, ColumnView<'b>>,
}


#[derive(Hash, Eq, PartialEq)]
struct Value((u64, i16, i8));


impl Value {
    fn new(val: f64) -> Value {
        Value(integer_decode(val))
    }
}
impl<'b> DataFrame<'b> {
    pub fn new(data: &'b Matrix,
               names: &'b Vec<&'b str>,
               data_map: HashMap<&'b str, ColumnView<'b>>)
               -> DataFrame<'b> {
        DataFrame {
            data: data,
            names: names,
            data_map: data_map,
        }

    }

    pub fn from_array(data: &'b Matrix, names: &'b Vec<&'b str>) -> Result<DataFrame<'b>, &'b str> {
        if data.cols() != names.len() {
            return Err("mismatched dimensions");
        }
        let data_map: HashMap<&'b str, ColumnView<'b>> =
            // names.iter().enumerate().map(|(x, y)| (*y, data.column(x))).collect();
            names.iter().zip(data.axis_iter(Axis(1))).map(|(x, y)| (*x, y)).collect();

        Ok(DataFrame::new(data, names, data_map))
    }

    fn get(&self, name: &'b str) -> Option<&ColumnView<'b>> {
        self.data_map.get(&name)
    }

    fn get_mut(&mut self, name: &'b str) -> Option<&mut ColumnView<'b>> {
        self.data_map.get_mut(&name)
    }

    pub fn inner_join(self, other: DataFrame<'b>, on: &'b str) -> Result<DataFrame<'b>, &'b str> {
        let h: HashMap<Value, usize> = self.data_map
            .get(on)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y), x))
            .collect();
        let j: HashMap<Value, usize> = other.data_map
            .get(on)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y), x))
            .collect();
        let mut vs = vec![];
        for row_key in h.keys() {
            match j.get(row_key) {
                None => continue,
                Some(v) => {
                    let p = self.data
                        .row(*h.get(row_key).unwrap())
                        .as_slice()
                        .unwrap()
                        .to_vec();
                    let o = other.data.row(*v).as_slice().unwrap().to_vec();
                    let chain = p.iter()
                        .chain(o.iter())
                        .collect::<Vec<&f64>>();
                    let array_chain = Array::from_shape_vec((1, chain.len()),
                                                            chain.iter().map(|x| **x).collect())
                        .unwrap();
                    vs.push(array_chain);

                }
            };
        }
        let views = vs.iter().map(|x| x.view()).collect::<Vec<MatrixView<'b>>>();
        let mut name_chain =
            self.names.iter().chain(other.names).map(|x| *x).collect::<Vec<&'b str>>();
        name_chain.sort();
        name_chain.dedup();
        let joined_matrix = &stack(Axis(0), &views[..]).unwrap();
        DataFrame::from_array(joined_matrix, &name_chain)
    }
}

// The following implementation requires a copy, which we don't want!
// Alternative solution to inserting is to mutate the original matrix, but this also requires a
// copy!
// fn insert(&mut self, name: &'a str, value: Column) {
//     match self.data.entry(name) {
//         Entry::Occupied(mut o) => {
//             o.get_mut();
//         }
//         Entry::Vacant(v) => {
//             v.insert(value.view());
//         }
//     };
// }




// parallelized join
// parallelized concatenation
// parallelized frequency counts
// index dataframe?
// sample rows
// find/select
// sort
// statistics (mean, median, stdev)
// print
