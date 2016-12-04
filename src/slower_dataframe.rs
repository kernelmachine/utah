use ndarray::{Axis, stack};
use std::collections::BTreeMap;
use helper::*;
use join::*;
use error::*;
use types::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SlowerDataFrame {
    data: BTreeMap<ColumnType, SlowerInnerType>,
    index: BTreeMap<IndexType, usize>,
}

impl SlowerDataFrame {
    pub fn new(data: Matrix<f64>) -> SlowerDataFrame {
        let data: Matrix<SlowerInnerType> = data.mapv(SlowerInnerType::from);

        let columns: BTreeMap<ColumnType, usize> = (0..data.shape()[1])
            .enumerate()
            .map(|(x, y)| (ColumnType::Str(x.to_string()), y))
            .collect();

        let index: BTreeMap<IndexType, usize> = (0..data.shape()[0])
            .enumerate()
            .map(|(x, y)| (IndexType::Str(x.to_string()), y))
            .collect();

        SlowerDataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }


    pub fn columns(mut self, columns: Vec<String>) -> Result<SlowerDataFrame> {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .zip((0..columns.len()))
            .map(|(x, y)| (ColumnType::from(x.to_string()), y))
            .collect();
        Ok(self)
    }

    pub fn index(mut self, index: Vec<String>) -> Result<SlowerDataFrame> {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index.iter()
            .zip((0..index.len()))
            .map(|(x, y)| (IndexType::from(x.to_string()), y))
            .collect();
        Ok(self)
    }

    pub fn names(&self) -> Vec<ColumnType> {
        self.columns.keys().map(|x| x.to_owned()).collect()
    }

    pub fn get(self, name: &str) -> Result<Column<SlowerInnerType>> {
        let name = ColumnType::from(name.to_string());
        match self.columns.get(&name) {
            Some(x) => Ok(self.data.column(*x).to_owned()),
            None => {
                match name {
                    ColumnType::Str(z) => {
                        return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
                    }
                    ColumnType::Date(z) => {
                        return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
                    }
                    ColumnType::Int(z) => {
                        return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
                    }
                }
            }
        }
    }


    pub fn insert_row(mut self, data: Matrix<f64>, index: &str) -> Result<SlowerDataFrame> {
        let index = IndexType::from(index.to_string());
        let data = data.mapv(SlowerInnerType::from);
        let end = self.index.len();
        self.index.insert(index, end);
        self.data = match stack(Axis(0), &[self.data.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err(ErrorKind::StackFail.into()),
        };
        Ok(self)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }


    pub fn insert_column(mut self, data: Matrix<f64>, name: &str) -> Result<SlowerDataFrame> {
        let name = ColumnType::from(name.to_string());
        let data = data.mapv(SlowerInnerType::from);
        let end = self.columns.len();
        self.columns.insert(name, end);
        self.data = match stack(Axis(1), &[self.data.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err(ErrorKind::StackFail.into()),
        };
        Ok(self)

    }

    pub fn drop_column(&mut self, names: &[&str]) -> Result<SlowerDataFrame> {
        let mut idxs = vec![];
        let names: Vec<ColumnType> =
            names.iter().map(|x| ColumnType::from(x.to_string())).collect();

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

        Ok(SlowerDataFrame {
            data: self.data.select(Axis(1), &to_keep[..]),
            columns: new_map.to_owned(),
            index: self.index.to_owned(),
        })
    }

    pub fn drop_row(&mut self, indexes: &[&str]) -> Result<SlowerDataFrame> {
        let mut idxs = vec![];
        let indexes: Vec<IndexType> =
            indexes.iter().map(|x| IndexType::from(x.to_string())).collect();

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
        Ok(SlowerDataFrame {
            data: self.data.select(Axis(0), &to_keep[..]),
            columns: self.columns.to_owned(),
            index: new_map.to_owned(),
        })
    }

    pub fn concat(&self, axis: Axis, other: &SlowerDataFrame) -> Result<SlowerDataFrame> {
        match axis {
            Axis(0) => {
                self.index
                    .keys()
                    .zip(self.data.axis_iter(Axis(0)))
                    .chain(other.index.keys().zip(other.data.iter()))
            }
            Axis(1) => {
                self.columns
                    .keys()
                    .zip(self.data.axis_iter(Axis(1)))
                    .chain(other.columns.keys().zip(other.data.iter()))
            }

        }
    }


    pub fn inner_join(&self, other: &SlowerDataFrame) -> Result<SlowerDataFrame> {

        let idxs: Vec<(IndexType, usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin,
                      self.index.clone().into_iter(),
                      other.index.clone())
                .collect();

        if idxs.len() == 0 {
            return Err(ErrorKind::NoCommonValues.into());
        }

        let new_matrix: Matrix<SlowerInnerType> = {
            let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
            let i2: Vec<usize> = idxs.iter().map(|&(_, _, y)| y.unwrap()).collect();
            let mat1 = self.data.select(Axis(0), &i1[..]);
            let mat2 = other.data.select(Axis(0), &i2[..]);

            match stack(Axis(1), &[mat1.view(), mat2.view()]) {
                Ok(z) => z,
                Err(_) => return Err(ErrorKind::StackFail.into()),
            }
        };

        Ok(SlowerDataFrame {
            data: new_matrix,
            columns: concat_column_maps(&self.columns, &other.columns),
            index: merge_maps(&self.index, &other.index),
        })
    }

    // pub fn sort_values(&self, by: &str, ascending: bool, axis: Axis) -> Result<SlowerDataFrame> {
    //     let column = self.get(by)?;
    //     let mut enum_col: Vec<(usize, &SlowerInnerType)> =
    //         column.iter().enumerate().sort_by(|a, &b| a.1.partial_cmp(b.1).unwrap());
    //     let indices: Vec<usize> = enum_col.iter().map(|x| x.0).collect();
    //     for index in indices {
    //         let pos = indices.index(index);
    //         if let Some(x) = self.index.get_mut(key) {
    //             *x = value;
    //         }
    //     }
    //
    //     self.data = self.data.select(Axis(0), indices.as_slice());
    //     Ok(*self)
    // }
}
