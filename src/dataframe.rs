
use ndarray::{Array, Ix, Axis, ArrayView, stack};
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

#[derive(Debug, PartialEq)]
pub struct DataFrame<'b> {
    data_map: HashMap<&'b str, ColumnView<'b>>,
}


#[derive(Hash, Eq, PartialEq)]
struct Value((u64, i16, i8));


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
    pub fn new(data_map: HashMap<&'b str, ColumnView<'b>>) -> DataFrame<'b> {
        DataFrame { data_map: data_map }

    }

    pub fn from_array(data: &'b Matrix, names: &'b Vec<&'b str>) -> Result<DataFrame<'b>, &'b str> {
        if data.cols() != names.len() {
            return Err("mismatched dimensions");
        }
        let data_map: HashMap<&'b str, ColumnView<'b>> =
            names.iter().zip(data.axis_iter(Axis(1))).map(|(x, y)| (*x, y)).collect();

        Ok(DataFrame::new(data_map))
    }

    pub fn from_vec(data: Vec<Column>, names: Vec<&'b str>) -> Result<DataFrame<'b>, &'b str> {
        if data.len() != names.len() {
            return Err("mismatched dimensions");
        }
        let data_map: HashMap<&'b str, Column> =
            names.iter().zip(data).map(|(x, y)| (*x, y.view())).collect();

        Ok(DataFrame::new(data_map))
        // let data = stack(Axis(0),
        //                  &data.iter()
        //                      .map(|x| x.view().into_shape((1, x.dim())).unwrap())
        //                      .collect::<Vec<MatrixView<'b>>>()[..])
        //     .unwrap();
        // if data.cols() != names.len() {
        //     return Err("mismatched dimensions");
        // }
        // let data_map: HashMap<&'b str, Column> =
        //     names.iter().zip(data.axis_iter(Axis(1))).map(|(x, y)| (*x, y.to_owned())).collect();
        //
        // Ok(DataFrame::new(data_map))
    }

    pub fn get(&self, name: &'b str) -> Option<&ColumnView<'b>> {
        self.data_map.get(&name)
    }

    // fn get_mut(&mut self, name: &'b str) -> Option<&mut Column> {
    //     self.data_map.get_mut(&name)
    // }

    pub fn inner_join(self, other: DataFrame<'b>, on: &'b str) -> Result<DataFrame<'b>, &'b str> {
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
            .map(|(_, y)| y.select(Axis(0), &idxs[..]));
        let o = other.data_map
            .iter()
            .filter(|&(x, _)| *x != on)
            .map(|(_, y)| y.select(Axis(0), &vs[..]));

        let data_zip: Vec<(Column, Column)> = p.zip(o).collect();
        let data_chain = data_zip.iter()
            .map(|&(ref x, ref y)| stack![Axis(0), *x, *y]);
        let name_chain = self.data_map
            .keys()
            .chain(other.data_map.keys().filter(|&x| *x != on))
            .map(|x| *x);
        DataFrame::from_vec(data_chain.collect(), name_chain.collect())
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
