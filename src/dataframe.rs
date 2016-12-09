use ndarray::{Axis, stack};
use std::collections::BTreeMap;
use helper::*;
use join::*;
use error::*;
use types::*;
use std::string::ToString;
use std::iter::{Iterator, FromIterator, Chain, Map, Filter};
use ndarray::{Elements, AxisIter};
use std::collections::btree_map::Iter;
use std::iter::Sum;
use itertools::PutBack;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    columns: BTreeMap<OuterType, usize>,
    data: Matrix<InnerType>,
    index: BTreeMap<OuterType, usize>,
}



pub enum DataFrameIterator<'a> {
    DataFrameRowIterator {
        index: Iter<'a, OuterType, usize>,
        axis_iter: AxisIter<'a, InnerType, usize>,
    },
    DataFrameColIterator {
        columns: Iter<'a, OuterType, usize>,
        axis_iter: AxisIter<'a, InnerType, usize>,
    },
}


impl<'a> Iterator for DataFrameIterator<'a> {
    type Item = (OuterType, RowView<'a, InnerType>);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            &mut DataFrameIterator::DataFrameColIterator { ref mut columns, ref mut axis_iter } => {
                match columns.next() {
                    Some((k, _)) => {
                        match axis_iter.next() {
                            Some(z) => return Some((k.clone(), z)),
                            None => None,
                        }
                    }
                    None => None,
                }
            }
            &mut DataFrameIterator::DataFrameRowIterator { ref mut index, ref mut axis_iter } => {
                match index.next() {
                    Some((k, _)) => {
                        match axis_iter.next() {
                            Some(z) => return Some((k.clone(), z)),
                            None => None,
                        }
                    }
                    None => None,
                }
            }
        }
    }
}


impl<'a> DataFrameIterator<'a> {
    pub fn select<T>(self, name: T) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a
        where OuterType: From<T>
    {
        let name = OuterType::from(name);
        self.filter(move |&(ref x, _)| *x == name)
    }

    pub fn concat(self, other : DataFrameIterator<'a>) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a


    {
        self.chain(other)
    }

    pub fn remove<T>(self, name: T) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a
        where OuterType: From<T>

    {
        let name = OuterType::from(name);
        self.filter(move |&(ref x, _)| *x != name)
    }

    pub fn add<T>(self, name: T, data : RowView<'a, InnerType>) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a
        where OuterType: From<T>

    {
        let name = OuterType::from(name);
        let mut it = PutBack::new(self);
        it.put_back((name,data));
        it
    }

    pub fn submatrix<T>(self, ind : &'a [OuterType]) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a
        where OuterType: From<T>{
            self.filter(move |&(ref x, _)| ind.contains(x))
    }




}
impl DataFrame {
    pub fn new<T: Clone>(data: Matrix<T>) -> DataFrame
        where InnerType: From<T>
    {
        let data: Matrix<InnerType> = data.mapv(InnerType::from);

        let columns: BTreeMap<OuterType, usize> = (0..data.shape()[1])
            .enumerate()
            .map(|(x, y)| (OuterType::Str(x.to_string()), y))
            .collect();

        let index: BTreeMap<OuterType, usize> = (0..data.shape()[0])
            .enumerate()
            .map(|(x, y)| (OuterType::Str(x.to_string()), y))
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }

    pub fn iter<'a>(&'a self, axis: Axis) -> Result<DataFrameIterator<'a>> {
        match axis {
            Axis(0) => {
                Ok(DataFrameIterator::DataFrameRowIterator {
                    index: self.index.iter(),
                    axis_iter: self.data.axis_iter(Axis(0)),
                })
            }
            Axis(1) => {
                Ok(DataFrameIterator::DataFrameColIterator {
                    columns: self.columns.iter(),
                    axis_iter: self.data.axis_iter(Axis(1)),
                })
            }
            _ => return Err(ErrorKind::InvalidAxis.into()),

        }


    }


    pub fn columns<'a, T>(mut self, columns: &'a [T]) -> Result<DataFrame>
        where OuterType: From<&'a T>
    {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .map(|x| OuterType::from(x))
            .zip((0..columns.len()))
            .collect();
        Ok(self)
    }

    pub fn index<'a, T>(mut self, index: &'a [T]) -> Result<DataFrame>
        where OuterType: From<&'a T>
    {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index.iter()
            .map(|x| OuterType::from(x))
            .zip((0..index.len()))
            .collect();
        Ok(self)
    }

    pub fn join<'a>(self, other :DataFrame) -> impl Iterator<Item = (OuterType,RowView<'a,InnerType>)> + 'a
    {

        let idxs: Vec<(OuterType , usize, Option<usize>)> =
            Join::new(JoinType::InnerJoin,
                      self.index.clone().into_iter(),
                      other.index.clone())
                .collect();
        let i1: Vec<OuterType> = idxs.iter().filter(|x| x.2.is_some()).map(|&(x, _, _)| x).collect();
        let df :  DataFrameIterator<'a> = self.iter(Axis(0)).unwrap().submatrix(&i1);
        let other_df : DataFrameIterator<'a> = other.iter(Axis(0)).unwrap().submatrix(&i1[..]);
        df.concat(other_df)

    }

    // pub fn names(&self) -> Vec<OuterType> {
    //     self.columns.keys().map(|x| x.to_owned()).collect()
    // }
    // pub fn chain<'a>(&'a self,
    //                  other: &'a DataFrame)
    //                  -> Chain<AxisIter<'a, InnerType, usize>, AxisIter<'a, InnerType, usize>> {
    //     self.data.axis_iter(Axis(0)).chain(other.data.axis_iter(Axis(0)))
    // }
    //
    //
    //
    // pub fn get_index<T>(self, name: T) -> Result<Row<InnerType>>
    //     where OuterType: From<T>
    // {
    //     let name = OuterType::from(name);
    //     match self.index.get(&name) {
    //         Some(x) => Ok(self.data.row(*x).to_owned()),
    //         None => {
    //             match name {
    //                 OuterType::Str(z) => {
    //                     return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
    //                 }
    //                 OuterType::Date(z) => {
    //                     return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
    //                 }
    //                 OuterType::Int(z) => {
    //                     return Err(ErrorKind::InvalidColumnName(z.to_string()).into())
    //                 }
    //             }
    //         }
    //     }
    // }
    //
    //
    //
    //
    // pub fn insert_row<T: Clone, S>(mut self, data: Matrix<T>, index: S) -> Result<DataFrame>
    //     where OuterType: From<S>,
    //           InnerType: From<T>
    // {
    //     let index = OuterType::from(index);
    //     let data = data.mapv(InnerType::from);
    //     let end = self.index.len();
    //     self.index.insert(index, end);
    //     self.data = match stack(Axis(0), &[self.data.view(), data.view()]) {
    //         Ok(z) => z,
    //         Err(_) => return Err(ErrorKind::StackFail.into()),
    //     };
    //     Ok(self)
    // }
    //
    // pub fn shape(&self) -> (usize, usize) {
    //     self.data.dim()
    // }
    //
    //
    // pub fn insert_column<T: Clone, S>(mut self, data: Matrix<T>, name: S) -> Result<DataFrame>
    //     where OuterType: From<S>,
    //           InnerType: From<T>
    // {
    //     let name = OuterType::from(name);
    //     let data = data.mapv(InnerType::from);
    //     let end = self.columns.len();
    //     self.columns.insert(name, end);
    //     self.data = match stack(Axis(1), &[self.data.view(), data.view()]) {
    //         Ok(z) => z,
    //         Err(_) => return Err(ErrorKind::StackFail.into()),
    //     };
    //     Ok(self)
    //
    // }
    //
    // pub fn drop_column<'a, T>(&mut self, names: &'a [T]) -> Result<DataFrame>
    //     where OuterType: From<&'a T>
    // {
    //     let mut idxs = vec![];
    //
    //     let new_map: &mut BTreeMap<OuterType, usize> = &mut self.columns.clone();
    //     for name in names.iter() {
    //         let name = OuterType::from(name);
    //         let idx = new_map.remove(&name);
    //         idxs.push(idx.unwrap());
    //         for (_, y) in new_map.iter_mut() {
    //             if y > &mut idx.unwrap() {
    //                 *y -= 1;
    //             }
    //         }
    //     }
    //     let cols = self.shape().1;
    //     let to_keep: Vec<usize> = (0..cols).filter(|x| !idxs.iter().any(|&y| y == *x)).collect();
    //
    //     Ok(DataFrame {
    //         data: self.data.select(Axis(1), &to_keep[..]),
    //         columns: new_map.to_owned(),
    //         index: self.index.to_owned(),
    //     })
    // }
    //
    // pub fn drop_row<'a, T>(&mut self, indexes: &'a [T]) -> Result<DataFrame>
    //     where OuterType: From<&'a T>
    // {
    //     let mut idxs = vec![];
    //
    //     let new_map: &mut BTreeMap<OuterType, usize> = &mut self.index.clone();
    //     for name in indexes.iter() {
    //         let name = OuterType::from(name);
    //         let idx = new_map.remove(&name);
    //         idxs.push(idx.unwrap());
    //         for (_, y) in new_map.iter_mut() {
    //             if y > &mut idx.unwrap() {
    //                 *y -= 1;
    //             }
    //         }
    //     }
    //     let rows = self.shape().0;
    //     let to_keep: Vec<usize> = (0..rows).filter(|x| !idxs.iter().any(|&y| y == *x)).collect();
    //     Ok(DataFrame {
    //         data: self.data.select(Axis(0), &to_keep[..]),
    //         columns: self.columns.to_owned(),
    //         index: new_map.to_owned(),
    //     })
    // }
    //
    // pub fn concat(&self, axis: Axis, other: &DataFrame) -> Result<DataFrame> {
    //
    //     match axis {
    //         Axis(0) => {
    //             if self.shape().1 == other.shape().1 {
    //                 let new_map: BTreeMap<OuterType, usize> = self.columns
    //                     .iter()
    //                     .map(|(x, y)| (x.to_owned(), *y))
    //                     .collect();
    //
    //                 let new_index: BTreeMap<OuterType, usize> = concat_index_maps(&self.index,
    //                                                                               &other.index);
    //
    //                 let new_matrix = match stack(Axis(0), &[self.data.view(), other.data.view()]) {
    //                     Ok(z) => z,
    //                     Err(_) => return Err(ErrorKind::StackFail.into()),
    //                 };
    //                 Ok(DataFrame {
    //                     data: new_matrix,
    //                     columns: new_map,
    //                     index: new_index,
    //                 })
    //             } else {
    //                 return Err(ErrorKind::ColumnShapeMismatch.into());
    //             }
    //         }
    //         Axis(1) => {
    //             if self.shape().0 == other.shape().0 {
    //                 let other_map: BTreeMap<OuterType, usize> = other.columns
    //                     .iter()
    //                     .map(|(x, y)| (x.to_owned(), y + self.columns.len()))
    //                     .collect();
    //                 let new_map: BTreeMap<OuterType, usize> = self.columns
    //                     .iter()
    //                     .chain(other_map.iter())
    //                     .map(|(x, y)| (x.to_owned(), *y))
    //                     .collect();
    //
    //                 let new_matrix = match stack(Axis(1), &[self.data.view(), other.data.view()]) {
    //                     Ok(z) => z,
    //                     Err(_) => return Err(ErrorKind::StackFail.into()),
    //                 };
    //                 Ok(DataFrame {
    //                     data: new_matrix,
    //                     columns: new_map,
    //                     index: self.index.to_owned(),
    //                 })
    //             } else {
    //                 return Err(ErrorKind::RowShapeMismatch.into());
    //             }
    //         }
    //         _ => return Err(ErrorKind::InvalidAxis.into()),
    //     }
    //
    // }
    //
    //
    // pub fn inner_join(&self, other: &DataFrame) -> Result<DataFrame> {
    //
    //     let idxs: Vec<(OuterType, usize, Option<usize>)> =
    //         Join::new(JoinType::InnerJoin,
    //                   self.index.clone().into_iter(),
    //                   other.index.clone())
    //             .collect();
    //
    //     if idxs.len() == 0 {
    //         return Err(ErrorKind::NoCommonValues.into());
    //     }
    //
    //     let new_matrix: Matrix<InnerType> = {
    //         let i1: Vec<usize> = idxs.iter().map(|&(_, x, _)| x).collect();
    //         let i2: Vec<usize> = idxs.iter().map(|&(_, _, y)| y.unwrap()).collect();
    //         let mat1 = self.data.select(Axis(0), &i1[..]);
    //         let mat2 = other.data.select(Axis(0), &i2[..]);
    //
    //         match stack(Axis(1), &[mat1.view(), mat2.view()]) {
    //             Ok(z) => z,
    //             Err(_) => return Err(ErrorKind::StackFail.into()),
    //         }
    //     };
    //
    //     Ok(DataFrame {
    //         data: new_matrix,
    //         columns: concat_column_maps(&self.columns, &other.columns),
    //         index: merge_maps(&self.index, &other.index),
    //     })
    // }

    // pub fn sort_values(&self, by: &str, ascending: bool, axis: Axis) -> Result<DataFrame> {
    //     let column = self.get(by)?;
    //     let mut enum_col: Vec<(usize, &InnerType)> =
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
