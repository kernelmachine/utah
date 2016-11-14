#[allow(dead_code)]


use ndarray::{Array, Ix, Axis, ArrayView, stack};
use std::collections::HashMap;
use std::mem;
use std::hash::Hash;
use std::fmt::Debug;

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
    Float(f64),
}

// impl Data {
//     fn from_int(i: i64) -> Data {
//         Data::Int(i)
//     }
//     fn from_float(f: f64) -> Data {
//         Data::Float(f)
//     }
//     fn from_string(s: String) -> Data {
//         Data::Str(s)
//     }asdfasdf
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



#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    data_map: HashMap<String, usize>,
    inner_matrix: Matrix,
}


impl DataFrame {
    pub fn inner_matrix(&self) -> &Matrix {
        &self.inner_matrix
    }
    pub fn datamap(&self) -> &HashMap<String, usize> {
        &self.data_map
    }
}

impl DataFrame {
    pub fn new(data: Matrix, names: Vec<String>) -> Result<DataFrame, &'static str> {

        if names.len() != data.shape()[1] {
            return Err("shape mismatch!");
        }


        let datamap: HashMap<String, usize> = names.iter()
            .enumerate()
            .map(|(x, y)| (y.clone(), x))
            .collect();

        let dm = DataFrame {
            data_map: datamap,
            inner_matrix: data,
        };

        Ok(dm)
    }



    pub fn inner_join(self, other: DataFrame, on: &str) -> Result<DataFrame, &'static str> {




        let h: HashMap<Value, usize> = {
            let h_idx: &usize = self.data_map
                .get(on)
                .unwrap();

            self.inner_matrix
                .column(*h_idx)
                .indexed_iter()
                .map(|(x, y)| (Value::new(*y).unwrap(), x))
                .collect()
        };

        let j: HashMap<Value, usize> = {
            let j_idx: &usize = other.data_map
                .get(on)
                .unwrap();

            other.inner_matrix
                .column(*j_idx)
                .indexed_iter()
                .map(|(x, y)| (Value::new(*y).unwrap(), x))
                .collect()
        };

        let idxs: Vec<(Value, usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin, h.into_iter(), j).collect();



        if idxs.len() == 0 {
            return Err("no common values");
        }
        let new_matrix: Matrix = {
            let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
            let i2: Vec<usize> =
                idxs.iter().filter(|&x| x.2.is_some()).map(|&(_, _, y)| y.unwrap()).collect();
            let mat1 = self.inner_matrix.select(Axis(0), &i1[..]);
            let mat2 = other.inner_matrix.select(Axis(0), &i2[..]);
            stack(Axis(1), &[mat1.view(), mat2.view()]).unwrap()
        };

        let mut n1: Vec<String> = self.data_map.keys().map(|x| x.clone()).collect();
        let n2: Vec<String> = other.data_map.keys().map(|x| x.clone() + "_x").collect();
        n1.extend_from_slice(&n2[..]);


        println!("join names shape - {:?}", n1.len());
        println!("join matrix shape - {:?}", new_matrix.shape()[1]);
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




pub enum JoinType {
    InnerJoin,
    OuterLeftJoin,
}
pub enum Join<L, K, RV> {
    InnerJoin { left: L, right: HashMap<K, RV> },
    OuterLeftJoin { left: L, right: HashMap<K, RV> },
}

impl<L, K, RV> Join<L, K, RV>
    where K: Hash + Eq
{
    pub fn new<LI, RI>(t: JoinType, left: LI, right: RI) -> Self
        where L: Iterator<Item = LI::Item>,
              LI: IntoIterator<IntoIter = L>,
              RI: IntoIterator<Item = (K, RV)>
    {
        match t {
            JoinType::InnerJoin => {
                Join::InnerJoin {
                    left: left.into_iter(),
                    right: right.into_iter().collect(),
                }
            }
            JoinType::OuterLeftJoin => {
                Join::OuterLeftJoin {
                    left: left.into_iter(),
                    right: right.into_iter().collect(),
                }
            }

        }

    }
}

impl<L, K, LV, RV> Iterator for Join<L, K, RV>
    where L: Iterator<Item = (K, LV)>,
          K: Hash + Eq + Debug,
          RV: Clone + Debug
{
    type Item = (K, LV, Option<RV>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            &mut Join::InnerJoin { ref mut left, ref right } => {
                loop {
                    match left.next() {
                        Some((k, lv)) => {
                            let rv = right.get(&k);
                            match rv {
                                Some(v) => return Some((k, lv, Some(v).cloned())),
                                None => continue,
                            }
                        }
                        None => return None,
                    }

                }
            }
            &mut Join::OuterLeftJoin { ref mut left, ref right } => {
                match left.next() {
                    Some((k, lv)) => {
                        let rv = right.get(&k);
                        Some((k, lv, rv.cloned()))
                    }
                    None => None,
                }
            }
        }

    }
}


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
