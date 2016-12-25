

use std::iter::Map;
use std::collections::btree_map::Values;
use std::slice::Iter;
use std::collections::BTreeMap;

pub trait MixedDataframeConstructor<'a, T, S>
    where   T: 'a + Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+One,
            S: 'a + Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>,
            Self : Sized
{
    fn new<U: Clone + Debug + Ord>(data: BTreeMap<U, Row<T>>) -> Self where S: From<U>, U : 'a;
    fn index<U: Clone + Ord>(self, index: &'a [U]) -> Result<Self> where S: From<U>;
    fn columns<U: Clone + Ord>(self, columns: &'a [U]) -> Result<Self> where S: From<U>;
    fn from_array(data: Row<T>, axis: UtahAxis) -> Self;
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, T, S>;
    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a, T, S>;
}
#[derive(Debug, Clone, PartialEq)]
pub struct MixedDataFrame<T, S>
    where T: Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Empty<T> + One,
          S: Hash + PartialOrd + Eq + Ord + From<String>
{
    pub data: BTreeMap<S, Row<T>>,
    pub index: BTreeMap<S, usize>,
}

impl <'a, T, S> MixedDataframeConstructor<'a, Iter<'a, RowView<'a, T>>, T, S> for MixedDataFrame<T, S>
    where
            T: 'a + Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>+ Empty<T>+One,
          S: 'a + Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug+ From<String>{
              fn new<U: Clone + Debug + Ord>(data: BTreeMap<U, Row<T>>) -> Self where S: From<U>, U : 'a{

                  let data: BTreeMap<S,Row<T>> = data.iter().map(|(z, y)| (z.clone().into(),y.clone().mapv_into(|x| {
                      if x.is_empty(){
                           return T::empty();
                      }
                      else{
                          return x;
                      }
                  }))).collect();
                  let row_size : usize = data.values().take(1).len();
                  let index: BTreeMap<S, usize> = (0..row_size).zip((0..row_size))
                      .map(|(x,y)| (x.to_string().into(), y))
                      .collect();
                  MixedDataFrame{
                      data : data,
                      index : index
                  }
              }
              fn index<U: Clone + Ord>(mut self, index: &'a [U]) -> Result<Self> where S: From<U>{
                  let index : Vec<S> = index.iter().map(|x| x.clone().into()).collect();
                  if index.len() != self.index.len() {
                      return Err(ErrorKind::ColumnShapeMismatch.into());
                  }
                  let new_index : BTreeMap<S, usize> = index.iter().zip((0..index.len())).map(|(x,y)| (x.clone(), y)).collect();


                  self.index = new_index;
                  Ok(self)
              }
              fn columns<U: Clone + Ord>(mut self, columns: &'a [U]) -> Result<Self> where S: From<U>{
                  let columns : Vec<S> = columns.iter().map(|x| x.clone().into()).collect();
                  if columns.len() != self.data.len() {
                      return Err(ErrorKind::ColumnShapeMismatch.into());
                  }
                  let new_data : BTreeMap<S, Row<T>> = columns.iter().zip(self.data.values()).map(|(x,y)| (x.clone(), y.clone())).collect();


                  self.data = new_data;
                  Ok(self)
              }
              fn from_array(data: Row<T>, axis: UtahAxis) -> Self {
                    let mut map : BTreeMap<S, Row<T>> = BTreeMap::new();
                    let mut index : BTreeMap<S, usize> = BTreeMap::new();

                    match axis {
                        UtahAxis::Column => {
                            for (x,y) in (0..data.len()).zip(0..data.len()){
                                index.insert(x.to_string().into(), y);
                            }
                            map.insert("1".to_string().into(), data);
                        },
                        UtahAxis::Row => {
                            for (x,y) in (0..data.len()).zip(data.iter()){
                                map.insert(x.to_string().into(), Array::from_vec(vec![y.clone()]));
                            }
                            index.insert("1".to_string().into(), 0);
                        }
                    }


                    MixedDataFrame{
                        data : map,
                        index : index
                    }
                }
              fn df_iter(self, axis: UtahAxis) -> DataFrameIterator<'a, Iter<'a, RowView<'a, T>>, T, S>{
                 match axis {
                   UtahAxis::Column => {
                       DataFrameIterator {
                           names: self.columns.iter(),
                           data: self.data,
                           other: self.index.to_owned(),
                           axis: UtahAxis::Column,
                       }
                   }
                   _ => panic!()
               }
           }
              fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a, T, S>{unimplemented!()}
}
