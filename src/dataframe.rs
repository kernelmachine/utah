use ndarray::{Array, Ix, Axis, ArrayView, stack};
use std::collections::BTreeMap;
use std::collections::HashMap;

use std::hash::Hash;
use std::fmt::Debug;
use num::Float;

pub type Column<T> = Array<T, Ix>;
pub type Matrix<T> = Array<T, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;






#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<T: Float> {
    pub data_map: BTreeMap<String, usize>,
    pub inner_matrix: Matrix<T>,
}

impl<T: Float> DataFrame<T> {
    pub fn inner_matrix(&self) -> &Matrix<T> {
        &self.inner_matrix
    }
    pub fn datamap(&self) -> &BTreeMap<String, usize> {
        &self.data_map
    }

    pub fn new(data: Matrix<T>,
               datamap: BTreeMap<String, usize>)
               -> Result<DataFrame<T>, &'static str> {

        if datamap.len() != data.shape()[1] {
            return Err("shape mismatch!");
        }




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

    pub fn shape(&self) -> (usize, usize) {
        self.inner_matrix.dim()
    }

    pub fn merge_maps<K: Hash + Eq + Copy, V: Copy>(first_context: &HashMap<K, V>,
                                                    second_context: &HashMap<K, V>)
                                                    -> HashMap<K, V> {
        let mut new_context = HashMap::new();
        for (key, value) in first_context.iter() {
            new_context.insert(*key, *value);
        }
        for (key, value) in second_context.iter() {
            new_context.insert(*key, *value);
        }
        new_context
    }


    pub fn drop(&mut self, names: &[String]) -> Result<DataFrame<T>, &'static str> {
        let mut idxs = vec![];

        let new_map: &mut BTreeMap<String, usize> = &mut self.data_map.clone();
        for name in names.iter() {
            let idx = new_map.remove(name);
            idxs.push(idx.unwrap());
            for (_, y) in new_map.iter_mut() {
                if y > &mut idx.unwrap() {
                    *y -= 1;

                }
            }
        }
        let cols = self.shape().1;
        let to_keep: Vec<usize> = (0..cols).filter(|x| !idxs.iter().any(|&y| y == *x)).collect();

        DataFrame::new(self.inner_matrix.select(Axis(1), &to_keep[..]),
                       new_map.to_owned())
    }

    pub fn concat(&self, axis: Axis, other: &DataFrame<T>) -> Result<DataFrame<T>, &'static str> {

        match axis {
            Axis(0) => {
                if self.shape().1 == other.shape().1 {
                    let new_map: BTreeMap<String, usize> = self.data_map
                        .iter()
                        .map(|(ref x, &y)| (x.to_string(), y))
                        .collect();
                    let other_idxs: Vec<usize> = other.data_map
                        .values()
                        .map(|x| *x)
                        .collect();
                    let other_matrix = other.inner_matrix.select(Axis(1), &other_idxs[..]);
                    let new_matrix =
                        match stack(Axis(0), &[self.inner_matrix.view(), other_matrix.view()]) {
                            Ok(z) => z,
                            Err(_) => return Err("could not insert into matrix."),
                        };
                    DataFrame::new(new_matrix, new_map)
                } else {
                    return Err("DataFrame column dimensions do not match.");
                }
            }
            Axis(1) => {
                if self.shape().0 == other.shape().0 {
                    let other_map: BTreeMap<String, usize> = other.data_map
                        .iter()
                        .map(|(x, y)| (x.to_string(), y + self.data_map.len()))
                        .collect();
                    let new_map: BTreeMap<String, usize> = self.data_map
                        .iter()
                        .chain(other_map.iter())
                        .map(|(x, y)| (x.to_string(), *y))
                        .collect();

                    let new_matrix = match stack(Axis(1),
                                                 &[self.inner_matrix.view(),
                                                   other.inner_matrix.view()]) {
                        Ok(z) => z,
                        Err(_) => return Err("could not insert into matrix."),
                    };
                    DataFrame::new(new_matrix, new_map)
                } else {
                    return Err("DataFrame row dimensions do not match.");
                }
            }
            _ => Err("invalid axis"),
        }

    }
    pub fn inner_join(&self, other: &DataFrame<T>, on: &str) -> Result<DataFrame<T>, &'static str> {

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
        let n2: BTreeMap<String, usize> = other.data_map
            .iter()
            .map(|(x, y)| (x.to_string() + "_x", *y + self.data_map.len()))
            .collect();


        let n1: BTreeMap<String, usize> = self.data_map
            .iter()
            .chain(n2.iter())
            .map(|(x, y)| (x.to_string(), *y))
            .collect();

        // println!("{:?}", n1);
        // println!("join names shape - {:?}", n1.len());
        // println!("join matrix shape - {:?}", new_matrix.shape()[1]);
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
