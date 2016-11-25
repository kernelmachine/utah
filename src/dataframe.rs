use ndarray::{Array, Ix, Axis, ArrayView, stack};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use std::string::ToString;
use chrono::*;

pub type Column<InnerType> = Array<InnerType, Ix>;
pub type Matrix<InnerType> = Array<InnerType, (Ix, Ix)>;
pub type ColumnView<'a, T> = ArrayView<'a, T, Ix>;
pub type MatrixView<'a, T> = ArrayView<'a, T, (Ix, Ix)>;


#[derive(Hash, Eq ,PartialOrd, PartialEq, Ord , Clone, Debug)]
pub enum ColumnType {
    Str(String),
    Date(DateTime<UTC>),
}

#[derive(Hash, PartialOrd, PartialEq, Eq , Ord , Clone,  Debug)]
pub enum IndexType {
    Str(String),
    Date(DateTime<UTC>),
}

#[derive(PartialOrd, PartialEq,  Clone, Debug, Copy)]
pub enum InnerType {
    Float(f64),
    Int(i64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    pub columns: BTreeMap<ColumnType, usize>,
    pub inner_matrix: Matrix<InnerType>,
    pub index: BTreeMap<IndexType, usize>,
}

impl InnerType {
    pub fn from_float(f: f64) -> InnerType {
        InnerType::Float(f)
    }
}


pub fn merge_maps(first_context: &BTreeMap<IndexType, usize>,
                  second_context: &BTreeMap<IndexType, usize>)
                  -> BTreeMap<IndexType, usize> {
    let mut new_context: BTreeMap<IndexType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    for (key, value) in second_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    new_context
}

pub fn concat_index_maps(first_context: &BTreeMap<IndexType, usize>,
                         second_context: &BTreeMap<IndexType, usize>)
                         -> BTreeMap<IndexType, usize> {
    let mut new_context: BTreeMap<IndexType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            IndexType::Str(z) => {
                new_context.insert(IndexType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            IndexType::Date(z) => {
                new_context.insert(IndexType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }

        };
    }
    new_context
}

pub fn concat_column_maps(first_context: &BTreeMap<ColumnType, usize>,
                          second_context: &BTreeMap<ColumnType, usize>)
                          -> BTreeMap<ColumnType, usize> {
    let mut new_context: BTreeMap<ColumnType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            ColumnType::Str(z) => {
                new_context.insert(ColumnType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            ColumnType::Date(z) => {
                new_context.insert(ColumnType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
        };
    }
    new_context
}



impl DataFrame {
    pub fn inner_matrix(&self) -> &Matrix<InnerType> {
        &self.inner_matrix
    }
    pub fn datamap(&self) -> &BTreeMap<ColumnType, usize> {
        &self.columns
    }

    pub fn new(data: Matrix<InnerType>,
               datamap: BTreeMap<ColumnType, usize>,
               index: Option<BTreeMap<IndexType, usize>>)
               -> Result<DataFrame, &'static str> {

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
                let b: BTreeMap<IndexType, usize> = (0..data.shape()[0])
                    .enumerate()
                    .map(|(x, y)| (IndexType::Str(x.to_string()), y))
                    .collect();
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

    pub fn get(self, name: ColumnType) -> Result<Column<InnerType>, &'static str> {
        match self.columns.get(&name) {
            Some(x) => Ok(self.inner_matrix.column(*x).to_owned()),
            None => Err("no such column exists"),
        }
    }


    pub fn insert_column(mut self,
                         data: Matrix<InnerType>,
                         name: ColumnType)
                         -> Result<DataFrame, &'static str> {

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

    pub fn drop_row(&mut self, indexes: &[IndexType]) -> Result<DataFrame, &'static str> {
        let mut idxs = vec![];

        let new_map: &mut BTreeMap<IndexType, usize> = &mut self.index.clone();
        for name in indexes.iter() {
            let idx = new_map.remove(name);
            idxs.push(idx.unwrap());
            for (_, y) in new_map.iter_mut() {
                if y > &mut idx.unwrap() {
                    *y -= 1;
                }
            }
        }
        let rows = self.shape().0;
        let to_keep: Vec<usize> = (0..rows).filter(|x| !idxs.iter().any(|&y| y == *x)).collect();
        DataFrame::new(self.inner_matrix.select(Axis(0), &to_keep[..]),
                       self.columns.to_owned(),
                       Some(new_map.to_owned()))
    }

    pub fn drop_column(&mut self, names: &[ColumnType]) -> Result<DataFrame, &'static str> {
        let mut idxs = vec![];

        let new_map: &mut BTreeMap<ColumnType, usize> = &mut self.columns.clone();
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

    pub fn concat(&self, axis: Axis, other: &DataFrame) -> Result<DataFrame, &'static str> {

        match axis {
            Axis(0) => {
                if self.shape().1 == other.shape().1 {
                    let new_map: BTreeMap<ColumnType, usize> = self.columns
                        .iter()
                        .map(|(x, y)| (x.to_owned(), *y))
                        .collect();

                    let new_index: BTreeMap<IndexType, usize> = concat_index_maps(&self.index,
                                                                                  &other.index);


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
                    let other_map: BTreeMap<ColumnType, usize> = other.columns
                        .iter()
                        .map(|(x, y)| (x.to_owned(), y + self.columns.len()))
                        .collect();
                    let new_map: BTreeMap<ColumnType, usize> = self.columns
                        .iter()
                        .chain(other_map.iter())
                        .map(|(x, y)| (x.to_owned(), *y))
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
    pub fn inner_join(&self, other: &DataFrame) -> Result<DataFrame, &'static str> {

        let idxs: Vec<(IndexType, usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin,
                      self.index.clone().into_iter(),
                      other.index.clone())
                .collect();

        if idxs.len() == 0 {
            return Err("no common values");
        }

        let new_matrix: Matrix<InnerType> = {
            let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
            let i2: Vec<usize> = idxs.iter().map(|&(_, _, y)| y.unwrap()).collect();
            let mat1 = self.inner_matrix.select(Axis(0), &i1[..]);
            let mat2 = other.inner_matrix.select(Axis(0), &i2[..]);
            match stack(Axis(1), &[mat1.view(), mat2.view()]) {
                Ok(z) => z,
                Err(_) => return Err("could not build joined matrix."),
            }
        };

        DataFrame::new(new_matrix,
                       concat_column_maps(&self.columns, &other.columns),
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
