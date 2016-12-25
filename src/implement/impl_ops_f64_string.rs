use util::types::*;
use std::iter::Iterator;
use adapters::aggregate::*;
use adapters::process::*;
use adapters::join::*;
use adapters::transform::*;
use util::types::UtahAxis;
use util::traits::*;
use dataframe::*;

impl<'a> Operations<'a, f64, String> for DataFrame<f64, String> {
    /// Get the dimensions of the dataframe.
    fn shape(self) -> (usize, usize) {
        self.data.dim()
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

    fn select<U: ?Sized>(&'a self,
                         names: &'a [&'a U],
                         axis: UtahAxis)
                         -> SelectIter<'a, f64, String>
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
    fn remove<U: ?Sized>(&'a self,
                         names: &'a [&'a U],
                         axis: UtahAxis)
                         -> RemoveIter<'a, f64, String>
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
    fn append<U: ?Sized>(&'a mut self,
                         name: &'a U,
                         data: RowView<'a, f64>,
                         axis: UtahAxis)
                         -> AppendIter<'a, f64, String>
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
                       -> InnerJoinIter<'a, f64, String> {
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
                       -> OuterJoinIter<'a, f64, String> {

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
                        -> InnerJoinIter<'a, f64, String> {
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
                        -> OuterJoinIter<'a, f64, String> {
        OuterJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }


    fn concat(&'a self,
              other: &'a DataFrame<f64, String>,
              axis: UtahAxis)
              -> ConcatIter<'a, f64, String> {
        match axis {
            UtahAxis::Row => {
                Concat::new(self.df_iter(UtahAxis::Row),
                            other.df_iter(UtahAxis::Row),
                            self.columns.clone(),
                            UtahAxis::Row)
            }
            UtahAxis::Column => {
                Concat::new(self.df_iter(UtahAxis::Column),
                            other.df_iter(UtahAxis::Column),
                            self.index.clone(),
                            UtahAxis::Column)
            }
        }
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
    fn sumdf(&'a mut self, axis: UtahAxis) -> SumIter<'a, f64, String> {
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
    fn map<F, B>(&'a mut self, f: F, axis: UtahAxis) -> MapDFIter<'a, f64, String, F, B>
        where F: Fn(&f64) -> B,
              for<'r> F: Fn(&f64) -> B
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
    fn mean(&'a mut self, axis: UtahAxis) -> MeanIter<'a, f64, String> {

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
    fn maxdf(&'a mut self, axis: UtahAxis) -> MaxIter<'a, f64, String> {

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
    fn min(&'a mut self, axis: UtahAxis) -> MinIter<'a, f64, String> {

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
    fn stdev(&'a self, axis: UtahAxis) -> StdevIter<'a, f64, String> {
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
              -> ImputeIter<'a, f64, String> {

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
