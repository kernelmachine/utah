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
    pub columns: BTreeMap<String, usize>,
    pub inner_matrix: Matrix<T>,
    pub index: BTreeMap<String, usize>,
}


pub fn merge_maps(first_context: &BTreeMap<String, usize>,
                  second_context: &BTreeMap<String, usize>)
                  -> BTreeMap<String, usize> {
    let mut new_context: BTreeMap<String, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    for (key, value) in second_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    new_context
}


impl<T: Float> DataFrame<T> {
    pub fn inner_matrix(&self) -> &Matrix<T> {
        &self.inner_matrix
    }
    pub fn datamap(&self) -> &BTreeMap<String, usize> {
        &self.columns
    }

    pub fn new(data: Matrix<T>,
               datamap: BTreeMap<String, usize>,
               index: Option<BTreeMap<String, usize>>)
               -> Result<DataFrame<T>, &'static str> {

        if datamap.len() != data.shape()[1] {
            return Err("column shape mismatch!");
        }


        let idx = match index {
            Some(z) => {
                if z.len() != data.shape()[0] {
                    return Err("row shape mismatch!");
                }
                z
            }
            None => {
                let b: BTreeMap<String, usize> =
                    (0..data.shape()[0]).enumerate().map(|(x, y)| (x.to_string(), y)).collect();
                b
            }
        };

        let dm = DataFrame {
            columns: datamap,
            inner_matrix: data,
            index: idx,
        };

        Ok(dm)
    }

    pub fn get(self, name: String) -> Result<Column<T>, &'static str> {
        match self.columns.get(&name) {
            Some(x) => Ok(self.inner_matrix.column(*x).to_owned()),
            None => Err("no such column exists"),
        }
    }


    pub fn insert_column(mut self,
                         data: Matrix<T>,
                         name: String)
                         -> Result<DataFrame<T>, &'static str> {

        let datamap_idx = {
            self.columns.len()
        };
        self.columns.insert(name, datamap_idx);
        self.inner_matrix = match stack(Axis(1), &[self.inner_matrix.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err("could not insert into matrix."),
        };
        Ok(self)

    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner_matrix.dim()
    }

    pub fn drop_row(&mut self, indexes: &[String]) -> Result<DataFrame<T>, &'static str> {
        let mut idxs = vec![];

        let new_map: &mut BTreeMap<String, usize> = &mut self.index.clone();
        for name in indexes.iter() {
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
                       self.columns.to_owned(),
                       Some(new_map.to_owned()))
    }

    pub fn drop_column(&mut self, names: &[String]) -> Result<DataFrame<T>, &'static str> {
        let mut idxs = vec![];

        let new_map: &mut BTreeMap<String, usize> = &mut self.columns.clone();
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
                       new_map.to_owned(),
                       Some(self.index.to_owned()))
    }

    pub fn concat(&self, axis: Axis, other: &DataFrame<T>) -> Result<DataFrame<T>, &'static str> {

        match axis {
            Axis(0) => {
                if self.shape().1 == other.shape().1 {
                    let new_map: BTreeMap<String, usize> = self.columns
                        .iter()
                        .map(|(ref x, &y)| (x.to_string(), y))
                        .collect();
                    let other_index: BTreeMap<String, usize> = other.index
                        .iter()
                        .map(|(x, y)| (x.to_string() + "_x", y + self.index.len()))
                        .collect();
                    let new_index: BTreeMap<String, usize> = self.index
                        .iter()
                        .chain(other_index.iter())
                        .map(|(x, y)| (x.to_string(), *y))
                        .collect();
                    // let other_matrix = other.inner_matrix.select(Axis(1), &other_index[..]);
                    let new_matrix = match stack(Axis(0),
                                                 &[self.inner_matrix.view(),
                                                   other.inner_matrix.view()]) {
                        Ok(z) => z,
                        Err(_) => return Err("could not insert into matrix."),
                    };
                    DataFrame::new(new_matrix, new_map, Some(new_index))
                } else {
                    return Err("DataFrame column dimensions do not match.");
                }
            }
            Axis(1) => {
                if self.shape().0 == other.shape().0 {
                    let other_map: BTreeMap<String, usize> = other.columns
                        .iter()
                        .map(|(x, y)| (x.to_string(), y + self.columns.len()))
                        .collect();
                    let new_map: BTreeMap<String, usize> = self.columns
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
                    DataFrame::new(new_matrix, new_map, Some(self.index.to_owned()))
                } else {
                    return Err("DataFrame row dimensions do not match.");
                }
            }
            _ => Err("invalid axis"),
        }

    }
    pub fn inner_join(&self, other: &DataFrame<T>) -> Result<DataFrame<T>, &'static str> {

        let idxs: Vec<(String, usize, Option<usize>)> = Join::new(JoinType::InnerJoin,
                                                                  self.index.clone().into_iter(),
                                                                  other.index.clone())
            .collect();
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
        let ns: BTreeMap<String, usize> = other.columns
            .iter()
            .map(|(x, y)| (x.to_string() + "_x", *y + self.columns.len()))
            .collect();


        let names_chain: BTreeMap<String, usize> = self.columns
            .iter()
            .chain(ns.iter())
            .map(|(x, y)| (x.to_string(), *y))
            .collect();

        DataFrame::new(new_matrix,
                       names_chain,
                       Some(merge_maps(&self.index, &other.index)))
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
