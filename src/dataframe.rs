use error::*;
use types::*;
use std::string::ToString;
use std::iter::Iterator;
use aggregate::*;
use transform::*;
use ndarray::Axis;
use types::UtahAxis;
use process::*;
use join::*;
use std::fmt::Debug;
use std::hash::Hash;
use traits::*;
use std::ops::{Add, Sub, Mul, Div};

/// A read-only dataframe.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame<T, S>
    where T: Clone + Debug + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub columns: Vec<S>,
    pub data: Matrix<T>,
    pub index: Vec<S>,
}

/// A read-write dataframe
#[derive(Debug, PartialEq)]
pub struct MutableDataFrame<'a, T, S>
    where T: Clone + Debug + 'a + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T>,
          S: Hash + PartialOrd + PartialEq + Eq + Ord + Clone + Debug
{
    pub columns: Vec<S>,
    pub data: MatrixMut<'a, T>,
    pub index: Vec<S>,
}

impl<'a> DataframeOps<'a, f64, String> for DataFrame<f64, String> {
    /// Create a new dataframe. The only required argument is data to populate the dataframe. The data's elements can be any of `InnerType`.
    /// By default, the columns and index of the dataframe are `["1", "2", "3"..."N"]`, where *N* is
    /// the number of columns (or rows) in the data.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a);
    /// ```
    ///
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[InnerType::Float(2.0), InnerType::Str("ak".into())],
    ///                [InnerType::Int32(6), InnerType::Int64(10)]]);
    /// let df = DataFrame::new(a);
    /// ```
    fn new<U: Clone + Debug>(data: Matrix<U>) -> DataFrame<f64, String>
        where f64: From<U>
    {

        let data: Matrix<f64> = data.mapv(f64::from);
        let columns: Vec<String> = (0..data.shape()[1])
            .map(|x| x.to_string())
            .collect();

        let index: Vec<String> = (0..data.shape()[0])
            .map(|x| x.to_string())
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }

    fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> DataFrame<f64, String>
        where f64: From<U>
    {
        let res_dim = match axis {
            UtahAxis::Column => (data.len(), 1),
            UtahAxis::Row => (1, data.len()),
        };
        let data: Matrix<f64> = data.into_shape(res_dim).unwrap().mapv(f64::from);

        let columns: Vec<String> = (0..res_dim.1)
            .map(|x| x.to_string())
            .collect();

        let index: Vec<String> = (0..res_dim.0)
            .map(|x| x.to_string())
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }
    /// Populate the dataframe with a set of columns. The column elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).columns(&["a", "b"]);
    /// df.is_ok();
    /// ```
    fn columns<U>(mut self, columns: &'a [U]) -> Result<DataFrame<f64, String>>
        where String: From<&'a U>
    {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .map(|x| String::from(x))
            .collect();
        Ok(self)
    }

    /// Populate the dataframe with an index. The index elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]);
    /// df.is_ok();
    /// ```
    ///
    /// You can also populate the dataframe with both column names and index names, like so:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]).columns(&["a", "b"]).unwrap();
    /// ```
    fn index<U>(mut self, index: &'a [U]) -> Result<DataFrame<f64, String>>
        where String: From<&'a U>
    {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index.iter()
            .map(|x| String::from(x))
            .collect();
        Ok(self)
    }

    /// Get the dimensions of the dataframe.
    fn shape(self) -> (usize, usize) {
        self.data.dim()
    }


    /// Return a dataframe iterator over the specified `UtahAxis`.
    ///
    /// The dataframe iterator yields a mutable view of a row or column of the dataframe for
    /// computation. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]).columns(&["a", "b"]).unwrap();
    /// ```
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, f64, String> {
        match axis {
            UtahAxis::Row => {
                DataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter(Axis(0)),
                    other: self.columns.clone(),
                    axis: UtahAxis::Row,
                }
            }
            UtahAxis::Column => {
                DataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter(Axis(1)),
                    other: self.index.to_owned(),
                    axis: UtahAxis::Column,
                }
            }
        }
    }

    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> MutableDataFrameIterator<'a, f64, String> {
        match axis {
            UtahAxis::Row => {
                MutableDataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter_mut(Axis(0)),
                    axis: UtahAxis::Row,
                    other: self.columns.clone(),
                }
            }
            UtahAxis::Column => {
                MutableDataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter_mut(Axis(1)),
                    axis: UtahAxis::Row,
                    other: self.index.clone(),
                }
            }
        }
    }
    /// Select rows or columns over the specified `UtahAxis`.
    ///
    /// The Select transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.select(&["a", "c"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```

    fn select<U>(&'a self,
                 names: &'a [U],
                 axis: UtahAxis)
                 -> Select<'a, f64, String, DataFrameIterator<'a, f64, String>>
        where String: From<&'a U>
    {
        let names: Vec<String> = names.iter().map(|x| String::from(x)).collect();
        match axis {
            UtahAxis::Row => {
                Select::new(self.df_iter(UtahAxis::Row),
                            names,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Select::new(self.df_iter(UtahAxis::Column),
                            names,
                            self.index.clone(),
                            UtahAxis::Column)
            }
        }
    }

    /// Remove rows or columns over the specified `UtahAxis`.
    ///
    /// The Remove transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn remove<U>(&'a self,
                 names: &'a [U],
                 axis: UtahAxis)
                 -> Remove<'a, DataFrameIterator<'a, f64, String>, f64, String>
        where String: From<&'a U>
    {
        let names: Vec<String> = names.iter().map(|x| String::from(x)).collect();
        match axis {
            UtahAxis::Row => {
                Remove::new(self.df_iter(UtahAxis::Row),
                            names,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Remove::new(self.df_iter(UtahAxis::Column),
                            names,
                            self.index.clone(),
                            UtahAxis::Column)
            }
        }
    }

    /// Append  a row or column along the specified `UtahAxis`.
    ///
    /// The Remove transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn append<U>(&'a mut self,
                 name: &'a U,
                 data: RowView<'a, f64>,
                 axis: UtahAxis)
                 -> Append<'a, DataFrameIterator<'a, f64, String>, f64, String>
        where String: From<&'a U>
    {
        let name: String = String::from(name);
        match axis {
            UtahAxis::Row => {
                Append::new(self.df_iter(UtahAxis::Row),
                            name,
                            data,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Append::new(self.df_iter(UtahAxis::Column),
                            name,
                            data,
                            self.index.clone(),
                            UtahAxis::Column)
            }

        }
    }


    /// Perform an inner left join between two dataframes along the specified `UtahAxis`.
    ///

    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn inner_left_join(&'a self,
                       other: &'a DataFrame<f64, String>)
                       -> InnerJoin<'a, DataFrameIterator<'a, f64, String>, f64, String> {
        InnerJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())
    }

    /// Perform an outer left join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn outer_left_join(&'a self,
                       other: &'a DataFrame<f64, String>)
                       -> OuterJoin<'a, DataFrameIterator<'a, f64, String>, f64, String> {

        OuterJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())




    }

    /// Perform an inner right join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn inner_right_join(&'a self,
                        other: &'a DataFrame<f64, String>)
                        -> InnerJoin<'a, DataFrameIterator<'a, f64, String>, f64, String> {
        InnerJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    /// Perform an outer right join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn outer_right_join(&'a self,
                        other: &'a DataFrame<f64, String>)
                        -> OuterJoin<'a, DataFrameIterator<'a, f64, String>, f64, String> {
        OuterJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }


    /// Sum along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn sumdf(&'a mut self,
             axis: UtahAxis)
             -> Sum<'a, DataFrameIterator<'a, f64, String>, f64, String> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Sum::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),
            UtahAxis::Column => Sum::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),

        }
    }

    /// Map a function along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn map<F, B>(&'a mut self,
                 f: F,
                 axis: UtahAxis)
                 -> MapDF<'a, f64, String, DataFrameIterator<'a, f64, String>, F, B>
        where F: Fn(&f64) -> B,
              for<'r> F: Fn(&InnerType) -> B
    {

        match axis {
            UtahAxis::Row => MapDF::new(self.df_iter(UtahAxis::Row), f, UtahAxis::Row),
            UtahAxis::Column => MapDF::new(self.df_iter(UtahAxis::Column), f, UtahAxis::Column),

        }
    }
    /// Get the average of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn mean(&'a mut self,
            axis: UtahAxis)
            -> Mean<'a, DataFrameIterator<'a, f64, String>, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Mean::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Mean::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }
    }

    /// Get the maximum of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```no_run
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn maxdf(&'a mut self,
             axis: UtahAxis)
             -> Max<'a, DataFrameIterator<'a, f64, String>, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Max::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Max::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }
    }

    /// Get the minimum of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn min(&'a mut self,
           axis: UtahAxis)
           -> Min<'a, DataFrameIterator<'a, f64, String>, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Min::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Min::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }

    }

    /// Get the standard deviation along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn stdev(&'a self,
             axis: UtahAxis)
             -> Stdev<'a, DataFrameIterator<'a, f64, String>, f64, String> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Stdev::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Stdev::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }


    }
    /// Get the standard deviation along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn impute(&'a mut self,
              strategy: ImputeStrategy,
              axis: UtahAxis)
              -> Impute<'a, MutableDataFrameIterator<'a, f64, String>, f64, String> {

        let index = self.index.clone();
        let columns = self.columns.clone();
        match axis {
            UtahAxis::Row => {
                Impute::new(self.df_iter_mut(UtahAxis::Row),
                            strategy,
                            columns,
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Impute::new(self.df_iter_mut(UtahAxis::Column),
                            strategy,
                            index,
                            UtahAxis::Column)
            }

        }
    }
}



impl<'a> MutableDataFrame<'a, InnerType, OuterType> {
    /// Create a new dataframe. The only required argument is data to populate the dataframe. The data's elements can be any of `InnerType`.
    /// By default, the columns and index of the dataframe are `["1", "2", "3"..."N"]`, where *N* is
    /// the number of columns (or rows) in the data.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a);
    /// ```
    ///
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[InnerType::Float(2.0), InnerType::Str("ak".into())],
    ///                [InnerType::Int32(6), InnerType::Int64(10)]]);
    /// let df = DataFrame::new(a);
    /// ```
    pub fn to_df(self) -> DataFrame<InnerType, OuterType> {
        let d = self.data.map(|x| (x.as_ref().clone()));
        DataFrame {
            data: d,
            columns: self.columns,
            index: self.index,
        }
    }
}
impl<'a> DataframeOps<'a, InnerType, OuterType> for DataFrame<InnerType, OuterType> {
    /// Create a new dataframe. The only required argument is data to populate the dataframe. The data's elements can be any of `InnerType`.
    /// By default, the columns and index of the dataframe are `["1", "2", "3"..."N"]`, where *N* is
    /// the number of columns (or rows) in the data.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a);
    /// ```
    ///
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[InnerType::Float(2.0), InnerType::Str("ak".into())],
    ///                [InnerType::Int32(6), InnerType::Int64(10)]]);
    /// let df = DataFrame::new(a);
    /// ```
    fn new<U: Clone>(data: Matrix<U>) -> DataFrame<InnerType, OuterType>
        where InnerType: From<U>
    {
        let data: Matrix<InnerType> = data.mapv(InnerType::from);
        let data: Matrix<InnerType> = data.mapv_into(|x| {
            match x {
                InnerType::Float(y) => {
                    if y.is_nan() {
                        return InnerType::Empty;
                    } else {
                        return x;
                    }
                }
                _ => return x,
            }

        });
        let columns: Vec<OuterType> = (0..data.shape()[1])
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        let index: Vec<OuterType> = (0..data.shape()[0])
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }

    fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> DataFrame<InnerType, OuterType>
        where InnerType: From<U>
    {
        let res_dim = match axis {
            UtahAxis::Column => (data.len(), 1),
            UtahAxis::Row => (1, data.len()),
        };
        let data: Matrix<InnerType> = data.into_shape(res_dim).unwrap().mapv(InnerType::from);
        let data: Matrix<InnerType> = data.mapv_into(|x| {
            match x {
                InnerType::Float(y) => {
                    if y.is_nan() {
                        return InnerType::Empty;
                    } else {
                        return x;
                    }
                }
                _ => return x,
            }

        });
        let columns: Vec<OuterType> = (0..res_dim.1)
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        let index: Vec<OuterType> = (0..res_dim.0)
            .map(|x| OuterType::Str(x.to_string()))
            .collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }
    /// Populate the dataframe with a set of columns. The column elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).columns(&["a", "b"]);
    /// df.is_ok();
    /// ```
    fn columns<U>(mut self, columns: &'a [U]) -> Result<DataFrame<InnerType, OuterType>>
        where OuterType: From<&'a U>
    {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Ok(self)
    }

    /// Populate the dataframe with an index. The index elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]);
    /// df.is_ok();
    /// ```
    ///
    /// You can also populate the dataframe with both column names and index names, like so:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]).columns(&["a", "b"]).unwrap();
    /// ```
    fn index<U>(mut self, index: &'a [U]) -> Result<DataFrame<InnerType, OuterType>>
        where OuterType: From<&'a U>
    {
        if index.len() != self.data.shape()[0] {
            return Err(ErrorKind::RowShapeMismatch.into());
        }
        self.index = index.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Ok(self)
    }

    /// Get the dimensions of the dataframe.
    fn shape(self) -> (usize, usize) {
        self.data.dim()
    }


    /// Return a dataframe iterator over the specified `UtahAxis`.
    ///
    /// The dataframe iterator yields a mutable view of a row or column of the dataframe for
    /// computation. Example:
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2]).columns(&["a", "b"]).unwrap();
    /// ```
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, InnerType, OuterType> {
        match axis {
            UtahAxis::Row => {
                DataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter(Axis(0)),
                    other: self.columns.clone(),
                    axis: UtahAxis::Row,
                }
            }
            UtahAxis::Column => {
                DataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter(Axis(1)),
                    other: self.index.to_owned(),
                    axis: UtahAxis::Column,
                }
            }
        }
    }

    fn df_iter_mut(&'a mut self,
                   axis: UtahAxis)
                   -> MutableDataFrameIterator<'a, InnerType, OuterType> {
        match axis {
            UtahAxis::Row => {
                MutableDataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter_mut(Axis(0)),
                    axis: UtahAxis::Row,
                    other: self.columns.clone(),
                }
            }
            UtahAxis::Column => {
                MutableDataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter_mut(Axis(1)),
                    axis: UtahAxis::Row,
                    other: self.index.clone(),
                }
            }
        }
    }
    /// Select rows or columns over the specified `UtahAxis`.
    ///
    /// The Select transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.select(&["a", "c"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```

    fn select<U>(&'a self,
                 names: &'a [U],
                 axis: UtahAxis)
                 -> Select<'a, InnerType, OuterType, DataFrameIterator<'a, InnerType, OuterType>>
        where OuterType: From<&'a U>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            UtahAxis::Row => {
                Select::new(self.df_iter(UtahAxis::Row),
                            names,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Select::new(self.df_iter(UtahAxis::Column),
                            names,
                            self.index.clone(),
                            UtahAxis::Column)
            }
        }
    }

    /// Remove rows or columns over the specified `UtahAxis`.
    ///
    /// The Remove transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn remove<U>(&'a self,
                 names: &'a [U],
                 axis: UtahAxis)
                 -> Remove<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType>
        where OuterType: From<&'a U>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            UtahAxis::Row => {
                Remove::new(self.df_iter(UtahAxis::Row),
                            names,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Remove::new(self.df_iter(UtahAxis::Column),
                            names,
                            self.index.clone(),
                            UtahAxis::Column)
            }
        }
    }

    /// Append  a row or column along the specified `UtahAxis`.
    ///
    /// The Remove transform adaptor yields a mutable view of a row or column of the dataframe for
    /// computation.
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn append<U>(&'a mut self,
                 name: &'a U,
                 data: RowView<'a, InnerType>,
                 axis: UtahAxis)
                 -> Append<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType>
        where OuterType: From<&'a U>
    {
        let name = OuterType::from(name);
        match axis {
            UtahAxis::Row => {
                Append::new(self.df_iter(UtahAxis::Row),
                            name,
                            data,
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Append::new(self.df_iter(UtahAxis::Column),
                            name,
                            data,
                            self.index.clone(),
                            UtahAxis::Column)
            }

        }
    }


    /// Perform an inner left join between two dataframes along the specified `UtahAxis`.
    ///

    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn inner_left_join
        (&'a self,
         other: &'a DataFrame<InnerType, OuterType>)
         -> InnerJoin<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {
        InnerJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())
    }

    /// Perform an outer left join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn outer_left_join
        (&'a self,
         other: &'a DataFrame<InnerType, OuterType>)
         -> OuterJoin<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {

        OuterJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())




    }

    /// Perform an inner right join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn inner_right_join
        (&'a self,
         other: &'a DataFrame<InnerType, OuterType>)
         -> InnerJoin<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {
        InnerJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    /// Perform an outer right join between two dataframes along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn outer_right_join
        (&'a self,
         other: &'a DataFrame<InnerType, OuterType>)
         -> OuterJoin<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {
        OuterJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }


    /// Sum along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn sumdf(&'a mut self,
             axis: UtahAxis)
             -> Sum<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Sum::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),
            UtahAxis::Column => Sum::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),

        }
    }

    /// Map a function along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn map<F, B>
        (&'a mut self,
         f: F,
         axis: UtahAxis)
         -> MapDF<'a, InnerType, OuterType, DataFrameIterator<'a, InnerType, OuterType>, F, B>
        where F: Fn(&InnerType) -> B
    {

        match axis {
            UtahAxis::Row => MapDF::new(self.df_iter(UtahAxis::Row), f, UtahAxis::Row),
            UtahAxis::Column => MapDF::new(self.df_iter(UtahAxis::Column), f, UtahAxis::Column),

        }
    }
    /// Get the average of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn mean(&'a mut self,
            axis: UtahAxis)
            -> Mean<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Mean::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Mean::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }
    }

    /// Get the maximum of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```no_run
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn maxdf(&'a mut self,
             axis: UtahAxis)
             -> Max<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Max::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Max::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }
    }

    /// Get the minimum of entries along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn min(&'a mut self,
           axis: UtahAxis)
           -> Min<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Min::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Min::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }

    }

    /// Get the standard deviation along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn stdev(&'a self,
             axis: UtahAxis)
             -> Stdev<'a, DataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Stdev::new(self.df_iter(UtahAxis::Row), columns, UtahAxis::Row),
            UtahAxis::Column => Stdev::new(self.df_iter(UtahAxis::Column), index, UtahAxis::Row),

        }


    }
    /// Get the standard deviation along the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use ndarray::arr2;
    /// use dataframe::DataFrame;
    ///
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&[1, 2, 3]).columns(&["a", "b"]).unwrap();
    /// for (idx, row) in df.remove(&["b"], UtahAxis::Column) {
    ///        assert_eq!(row, a.row(idx))
    ///    }
    /// ```
    fn impute
        (&'a mut self,
         strategy: ImputeStrategy,
         axis: UtahAxis)
         -> Impute<'a, MutableDataFrameIterator<'a, InnerType, OuterType>, InnerType, OuterType> {

        let index = self.index.clone();
        let columns = self.columns.clone();
        match axis {
            UtahAxis::Row => {
                Impute::new(self.df_iter_mut(UtahAxis::Row),
                            strategy,
                            columns,
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Impute::new(self.df_iter_mut(UtahAxis::Column),
                            strategy,
                            index,
                            UtahAxis::Column)
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
