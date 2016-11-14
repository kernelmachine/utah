use ndarray::{Array, Ix, Axis, ArrayView, stack};
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use num::Float;

pub type Column<T> = Array<T, Ix>;
pub type Row<T> = Vec<T>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;






#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<T: Float> {
    data_map: HashMap<String, usize>,
    inner_matrix: Matrix<T>,
}


impl<T: Float> DataFrame<T> {
    pub fn inner_matrix(&self) -> &Matrix<T> {
        &self.inner_matrix
    }
    pub fn datamap(&self) -> &HashMap<String, usize> {
        &self.data_map
    }

    pub fn new(data: Matrix<T>, names: Vec<String>) -> Result<DataFrame<T>, &'static str> {

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

    pub fn get(self, name: String) -> Result<Column<T>, &'static str> {
        match self.data_map.get(&name) {
            Some(x) => Ok(self.inner_matrix.column(*x).to_owned()),
            None => Err("no such column exists"),
        }
    }


    pub fn insert(mut self, data: Matrix<T>, name: String) -> Result<DataFrame<T>, &'static str> {

        let idx = {
            self.data_map.len()
        };
        self.data_map.insert(name, idx);
        self.inner_matrix = match stack(Axis(1), &[self.inner_matrix.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err("could not insert into matrix."),
        };
        Ok(self)

    }

    pub fn inner_join(self, other: DataFrame<T>, on: &str) -> Result<DataFrame<T>, &'static str> {

        let h: HashMap<Value, usize> = {
            let h_idx: &usize = match self.data_map
                .get(on) {
                Some(z) => z,
                None => return Err("column name does not exist in self dataframe"),
            };

            self.inner_matrix
                .column(*h_idx)
                .indexed_iter()
                .map(|(x, y)| (Value::new(*y), x))
                .collect()
        };

        let j: HashMap<Value, usize> = {
            let j_idx: &usize = match other.data_map
                .get(on) {
                Some(z) => z,
                None => return Err("column name does not exist in other dataframe"),
            };

            other.inner_matrix
                .column(*j_idx)
                .indexed_iter()
                .map(|(x, y)| (Value::new(*y), x))
                .collect()
        };

        let idxs: Vec<(Value, usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin, h.into_iter(), j).collect();

        if idxs.len() == 0 {
            return Err("no common values");
        }

        let new_matrix: Matrix<T> = {
            let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
            let i2: Vec<usize> = idxs.iter().map(|&(_, _, y)| y.unwrap()).collect();
            let mat1 = self.inner_matrix.select(Axis(0), &i1[..]);
            let mat2 = other.inner_matrix.select(Axis(0), &i2[..]);
            match stack(Axis(1), &[mat1.view(), mat2.view()]) {
                Ok(z) => z,
                Err(_) => return Err("could not build joined matrix."),
            }

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
    fn new<T: Float>(val: T) -> Value {
        Value(val.integer_decode())
    }
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

// To implement....?
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
