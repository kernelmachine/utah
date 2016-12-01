use ndarray::{Axis, stack};
use std::collections::BTreeMap;
use helper::*;
use join::*;
use error::*;
use types::*;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    columns: BTreeMap<ColumnType, usize>,
    data: Matrix<InnerType>,
    index: BTreeMap<IndexType, usize>,
}

impl DataFrame {
    pub fn new(data: Matrix<f64>) -> DataFrame {
        let data: Matrix<InnerType> = data.mapv(InnerType::from);

        let columns: BTreeMap<ColumnType, usize> = (0..data.shape()[1])
            .enumerate()
            .map(|(x, y)| (ColumnType::Str(x.to_string()), y))
            .collect();

        let index: BTreeMap<IndexType, usize> = (0..data.shape()[0])
            .enumerate()
            .map(|(x, y)| (IndexType::Str(x.to_string()), y))
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }


    pub fn columns(mut self, columns: BTreeMap<ColumnType, usize>) -> Result<DataFrame> {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns;
        Ok(self)
    }

    pub fn index(mut self, index: BTreeMap<IndexType, usize>) -> Result<DataFrame> {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index;
        Ok(self)
    }

    pub fn names(&self) -> Vec<ColumnType> {
        self.columns.keys().map(|x| x.to_owned()).collect()
    }

    // pub fn new(data: Matrix<f64>,
    //            columns: BTreeMap<ColumnType, usize>,
    //            index: Option<BTreeMap<IndexType, usize>>)
    //            -> Result<DataFrame> {
    //
    //     let data: Matrix<InnerType> = data.mapv(InnerType::from);
    //
    //     if columns.len() != data.shape()[1] {
    //         return Err(ErrorKind::ColumnShapeMismatch.into());
    //     }
    //
    //
    //     let idx = match index {
    //         Some(z) => {
    //             if z.len() != data.shape()[0] {
    //                 return Err(ErrorKind::RowShapeMismatch.into());
    //             }
    //             z
    //         }
    //         None => {
    //             let b: BTreeMap<IndexType, usize> = (0..data.shape()[0])
    //                 .enumerate()
    //                 .map(|(x, y)| (IndexType::Str(x.to_string()), y))
    //                 .collect();
    //             b
    //         }
    //     };
    //
    //     let dm = DataFrame {
    //         columns: columns,
    //         data: data,
    //         index: idx,
    //     };
    //
    //     Ok(dm)
    // }

    pub fn get(self, name: ColumnType) -> Result<Column<InnerType>> {
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


    pub fn insert_row(mut self, data: Matrix<InnerType>, index: IndexType) -> Result<DataFrame> {

        let ind_idx = {
            self.index.len()
        };
        self.index.insert(index, ind_idx);
        self.data = match stack(Axis(0), &[self.data.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err(ErrorKind::StackFail.into()),
        };
        Ok(self)

    }

    pub fn insert_column(mut self, data: Matrix<f64>, name: ColumnType) -> Result<DataFrame> {

        let data = data.mapv(InnerType::from);
        let columns_idx = {
            self.columns.len()
        };
        self.columns.insert(name, columns_idx);
        self.data = match stack(Axis(1), &[self.data.view(), data.view()]) {
            Ok(z) => z,
            Err(_) => return Err(ErrorKind::StackFail.into()),
        };
        Ok(self)

    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn drop_row(&mut self, indexes: &[IndexType]) -> Result<DataFrame> {
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
        Ok(DataFrame {
            data: self.data.select(Axis(0), &to_keep[..]),
            columns: self.columns.to_owned(),
            index: new_map.to_owned(),
        })
    }

    pub fn drop_column(&mut self, names: &[ColumnType]) -> Result<DataFrame> {
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

        Ok(DataFrame {
            data: self.data.select(Axis(1), &to_keep[..]),
            columns: new_map.to_owned(),
            index: self.index.to_owned(),
        })
    }

    pub fn concat(&self, axis: Axis, other: &DataFrame) -> Result<DataFrame> {

        match axis {
            Axis(0) => {
                if self.shape().1 == other.shape().1 {
                    let new_map: BTreeMap<ColumnType, usize> = self.columns
                        .iter()
                        .map(|(x, y)| (x.to_owned(), *y))
                        .collect();

                    let new_index: BTreeMap<IndexType, usize> = concat_index_maps(&self.index,
                                                                                  &other.index);

                    let new_matrix = match stack(Axis(0), &[self.data.view(), other.data.view()]) {
                        Ok(z) => z,
                        Err(_) => return Err(ErrorKind::StackFail.into()),
                    };
                    Ok(DataFrame {
                        data: new_matrix,
                        columns: new_map,
                        index: new_index,
                    })
                } else {
                    return Err(ErrorKind::ColumnShapeMismatch.into());
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

                    let new_matrix = match stack(Axis(1), &[self.data.view(), other.data.view()]) {
                        Ok(z) => z,
                        Err(_) => return Err(ErrorKind::StackFail.into()),
                    };
                    Ok(DataFrame {
                        data: new_matrix,
                        columns: new_map,
                        index: self.index.to_owned(),
                    })
                } else {
                    return Err(ErrorKind::RowShapeMismatch.into());
                }
            }
            _ => return Err(ErrorKind::InvalidAxis.into()),
        }

    }
    pub fn inner_join(&self, other: &DataFrame) -> Result<DataFrame> {

        let idxs: Vec<(IndexType, usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin,
                      self.index.clone().into_iter(),
                      other.index.clone())
                .collect();

        if idxs.len() == 0 {
            return Err(ErrorKind::NoCommonValues.into());
        }

        let new_matrix: Matrix<InnerType> = {
            let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
            let i2: Vec<usize> = idxs.iter().map(|&(_, _, y)| y.unwrap()).collect();
            let mat1 = self.data.select(Axis(0), &i1[..]);
            let mat2 = other.data.select(Axis(0), &i2[..]);
            match stack(Axis(1), &[mat1.view(), mat2.view()]) {
                Ok(z) => z,
                Err(_) => return Err(ErrorKind::StackFail.into()),
            }
        };

        Ok(DataFrame {
            data: new_matrix,
            columns: concat_column_maps(&self.columns, &other.columns),
            index: merge_maps(&self.index, &other.index),
        })
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
