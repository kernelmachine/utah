use ndarray::{Array, Ix, Axis, ArrayView};
use std::collections::HashMap;
use std::mem;


pub type Column = Array<f64, Ix>;
pub type Row = Array<f64, Ix>;

pub type Matrix = Array<f64, (Ix, Ix)>;
pub type ColumnView<'a> = ArrayView<'a, f64, Ix>;
pub type MatrixView<'a> = ArrayView<'a, f64, (Ix, Ix)>;

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

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<'b> {
    data_map: HashMap<&'b str, Column>,
}


#[derive(Hash, Eq,  Clone,PartialEq)]
struct Value((u64, i16, i8));

pub fn merge<'b>(first_context: HashMap<&'b str, Column>,
                 second_context: HashMap<&'b str, Column>)
                 -> HashMap<&'b str, Column> {
    let mut new_context = HashMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(*key, value.to_owned());
    }
    for (key, value) in second_context.iter() {
        new_context.insert(*key, value.to_owned());
    }
    new_context
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

impl Value {
    fn new(val: f64) -> Value {
        Value(integer_decode(val))
    }
}


impl<'b> DataFrame<'b> {
    pub fn new(data_map: HashMap<&'b str, Column>) -> DataFrame<'b> {
        DataFrame { data_map: data_map }

    }

    pub fn from_array(data: &'b Matrix, names: &'b Vec<&'b str>) -> Result<DataFrame<'b>, &'b str> {
        if data.cols() != names.len() {
            return Err("mismatched dimensions");
        }
        let data_map: HashMap<&'b str, Column> =
            names.iter().zip(data.axis_iter(Axis(1))).map(|(x, y)| (*x, y.to_owned())).collect();

        Ok(DataFrame::new(data_map))
    }


    pub fn get(&self, name: &'b str) -> Option<&Column> {
        self.data_map.get(&name)
    }



    pub fn inner_join(&self, other: &DataFrame<'b>, on: &'b str) -> Result<DataFrame<'b>, &'b str> {
        let h: HashMap<Value, usize> = self.get(on)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y), x))
            .collect();
        let j: HashMap<Value, usize> = other.get(on)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(x, y)| (Value::new(*y), x))
            .collect();
        let mut idxs = vec![];
        let mut vs = vec![];
        for row_key in h.keys() {
            match j.get(row_key) {
                None => continue,
                Some(v) => {
                    let idx = *h.get(row_key).unwrap();
                    idxs.push(idx);
                    vs.push(*v);
                }
            };
        }
        let p = self.data_map
            .iter()
            .map(|(x, y)| (*x, y.select(Axis(0), &idxs[..])))
            .collect();
        let o = other.data_map
            .iter()
            .map(|(x, y)| (*x, y.select(Axis(0), &vs[..])))
            .collect();
        let res = merge(p, o);

        Ok(DataFrame::new(res))
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

// statistics (mean, median, stdev)
// print
