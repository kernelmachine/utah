use util::error::*;
use util::types::*;
use std::string::ToString;
use std::iter::Iterator;
use ndarray::Axis;
use util::traits::*;
use dataframe::*;

impl<'a, T> Constructor<'a, T> for DataFrame<T>
    where T: UtahNum + 'a
{
    /// Create a new dataframe. The only required argument is data to populate the dataframe.
    /// By default, the columns and index of the dataframe are `["1", "2", "3"..."N"]`, where *N* is
    /// the number of columns (or rows) in the data.
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df : DataFrame<f64> = DataFrame::new(a);
    /// ```
    ///
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[InnerType::Float(2.0), InnerType::Str("ak".into())],
    ///                [InnerType::Int32(6), InnerType::Int64(10)]]);
    /// let df : DataFrame<InnerType> = DataFrame::new(a);
    /// ```
    fn new<U: Clone>(data: Matrix<U>) -> DataFrame<T>
        where T: From<U>
    {
        let mut data: Matrix<T> = data.mapv(T::from);
        data.mapv_inplace(|x| {
            if x.is_empty() {
                return T::empty();
            } else {
                return x;
            }
        });

        let columns: Vec<String> = (0..data.shape()[1]).map(|x| x.to_string()).collect();

        let index: Vec<String> = (0..data.shape()[0]).map(|x| x.to_string()).collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }
    /// Generate a 1-dimensional DataFrame from an 1-D array of data.
    /// When populating the dataframe with mixed-types, wrap the elements with `InnerType` enum.
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr1(&[2.0, 7.0]);
    /// let df : DataFrame<f64> = DataFrame::from_array(a, UtahAxis::Column);
    /// ```
    ///
    fn from_array<U: Clone>(data: Row<U>, axis: UtahAxis) -> DataFrame<T>
        where T: From<U>
    {
        let res_dim = match axis {
            UtahAxis::Column => (data.len(), 1),
            UtahAxis::Row => (1, data.len()),
        };
        let data: Matrix<T> = data.into_shape(res_dim).unwrap().mapv(T::from);
        let data: Matrix<T> = data.mapv_into(|x| {
            if x.is_empty() {
                return T::empty();
            } else {
                return x;
            }
        });
        let columns: Vec<String> = (0..res_dim.1).map(|x| x.to_string()).collect();

        let index: Vec<String> = (0..res_dim.0).map(|x| x.to_string()).collect();

        DataFrame {
            data: data,
            columns: columns,
            index: index,
        }
    }
    /// Populate the dataframe with a set of columns. The column elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df : Result<DataFrame<f64>> = DataFrame::new(a).columns(&["a", "b"]);
    /// df.is_ok();
    /// ```
    fn columns<U: Clone>(mut self, columns: &'a [U]) -> Result<DataFrame<T>>
        where String: From<U>
    {
        let data_shape = self.data.shape()[1];
        let column_shape = columns.len();
        if column_shape != data_shape {
            return Err(ErrorKind::ColumnShapeMismatch(data_shape.to_string(),
                                                      column_shape.to_string())
                .into());
        }
        let new_columns: Vec<String> = columns.iter()
            .map(|x| x.clone().into())
            .collect();
        self.columns = new_columns;
        Ok(self)
    }

    /// Populate the dataframe with an index. The index elements can be any of `OuterType`. Example:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df : Result<DataFrame<f64>> = DataFrame::new(a).index(&["1", "2"]);
    /// df.is_ok();
    /// ```
    ///
    /// You can also populate the dataframe with both column names and index names, like so:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df : Result<DataFrame<f64>> = DataFrame::new(a).index(&["1", "2"]).unwrap().columns(&["a", "b"]);
    /// df.is_ok();
    /// ```
    fn index<U: Clone>(mut self, index: &'a [U]) -> Result<DataFrame<T>>
        where String: From<U>
    {
        let data_shape = self.data.shape()[0];
        let index_shape = index.len();
        if index_shape != data_shape {
            return Err(ErrorKind::IndexShapeMismatch(data_shape.to_string(),
                                                     index_shape.to_string())
                .into());
        }
        let new_index: Vec<String> = index.iter()
            .map(|x| x.clone().into())
            .collect();
        self.index = new_index;
        Ok(self)
    }


    /// Return a dataframe iterator over the specified `UtahAxis`.
    ///
    /// The dataframe iterator yields a view of a row or column of the dataframe for eventual
    /// processing. Example:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df : DataFrame<f64> = DataFrame::new(a).index(&["1", "2"]).unwrap().columns(&["a", "b"]).unwrap();
    /// let df_iter = df.df_iter(UtahAxis::Row);
    /// ```
    fn df_iter(&'a self, axis: UtahAxis) -> DataFrameIterator<'a, T> {
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

    /// Return a mutable dataframe iterator over the specified `UtahAxis`.
    ///
    /// The mutable dataframe iterator yields a view of a row or column of the dataframe for eventual
    /// processing. Example:
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let mut df : DataFrame<f64> = DataFrame::new(a);
    /// let df_iter_mut = df.df_iter_mut(UtahAxis::Column);
    /// ```
    fn df_iter_mut(&'a mut self, axis: UtahAxis) -> DataFrameMutIterator<'a, T> {
        match axis {
            UtahAxis::Row => {
                DataFrameMutIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter_mut(Axis(0)),
                    axis: UtahAxis::Row,
                    other: self.columns.clone(),
                }
            }
            UtahAxis::Column => {
                DataFrameMutIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter_mut(Axis(1)),
                    axis: UtahAxis::Row,
                    other: self.index.clone(),
                }
            }
        }
    }
}
