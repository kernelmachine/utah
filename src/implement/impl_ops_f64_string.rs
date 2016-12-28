use util::types::*;
use std::iter::Iterator;
use combinators::aggregate::*;
use combinators::process::*;
use combinators::interact::*;
use combinators::transform::*;
use util::types::UtahAxis;
use util::traits::*;
use dataframe::*;

impl<'a> Operations<'a, f64, String> for DataFrame<f64, String> {
    /// Get the dimensions of the dataframe.
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
    /// let df = DataFrame::new(a).index(&["1", "2"]).unwrap().columns(&["a", "b"]).unwrap();
    /// assert_eq!(df.shape(), (2,2));
    /// ```
    ///
    fn shape(self) -> (usize, usize) {
        self.data.dim()
    }


    /// Select rows or columns over the specified `UtahAxis`.
    ///
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0, 8.0], [3.0, 4.0, 9.0], [2.0, 8.0, 1.0]]);
    /// let df = DataFrame::new(a).index(&["1", "2", "3"]).unwrap().columns(&["a", "b", "c"]).unwrap();
    /// let res = df.select(&["a", "c"], UtahAxis::Column).as_df();
    /// res.is_ok();
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
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let df = DataFrame::new(a).index(&["1", "2", "3"]).unwrap().columns(&["a", "b"]).unwrap();
    /// let res = df.remove(&["a"], UtahAxis::Column).as_df();
    /// res.is_ok();
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
    ///
    /// ```
    /// use utah::prelude::*;
    /// let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
    /// let mut df = DataFrame::new(a).index(&["1", "2", "3"]).unwrap().columns(&["a", "b"]).unwrap();
    /// let new_data = df.select(&["2"], UtahAxis::Row).as_array().unwrap();
    /// let new_df =  df.append("4", new_data.view(), UtahAxis::Row).as_df();
    /// new_df.is_ok();
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
    fn inner_left_join(&'a self,
                       other: &'a DataFrame<f64, String>)
                       -> InnerJoinIter<'a, f64, String> {
        InnerJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())
    }

    /// Perform an outer left join between two dataframes along the specified `UtahAxis`.
    fn outer_left_join(&'a self,
                       other: &'a DataFrame<f64, String>)
                       -> OuterJoinIter<'a, f64, String> {

        OuterJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())




    }

    /// Perform an inner right join between two dataframes along the specified `UtahAxis`.

    fn inner_right_join(&'a self,
                        other: &'a DataFrame<f64, String>)
                        -> InnerJoinIter<'a, f64, String> {
        InnerJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    /// Perform an outer right join between two dataframes along the specified `UtahAxis`.
    fn outer_right_join(&'a self,
                        other: &'a DataFrame<f64, String>)
                        -> OuterJoinIter<'a, f64, String> {
        OuterJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    /// Concatenate two dataframes along the specified `UtahAxis`
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
    /// use utah::prelude::*;
    ///
    /// let a = arr2(&[[2., 6.], [3., 4.]]);
    /// let mut df: DataFrame<f64, String> = DataFrame::new(a).columns(&["a", "b"]).unwrap();
    /// let z: Result<DataFrame<f64, String>> = df.sumdf(UtahAxis::Row).as_df();
    /// ```
    fn sumdf(&'a mut self, axis: UtahAxis) -> SumIter<'a, f64, String> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Sum::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Sum::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }
    }

    /// Map a function along the specified `UtahAxis`.
    fn map<F>(&'a mut self, f: F, axis: UtahAxis) -> MapDFIter<'a, f64, String, F>
        where F: Fn(f64) -> f64,
              for<'r> F: Fn(f64) -> f64
    {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => MapDF::new(self.df_iter_mut(UtahAxis::Row), f, columns, UtahAxis::Row),
            UtahAxis::Column => {
                MapDF::new(self.df_iter_mut(UtahAxis::Column),
                           f,
                           index,
                           UtahAxis::Column)
            }

        }
    }
    /// Get the average of entries along the specified `UtahAxis`.

    fn mean(&'a mut self, axis: UtahAxis) -> MeanIter<'a, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Mean::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => {
                Mean::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column)
            }

        }
    }

    /// Get the maximum of entries along the specified `UtahAxis`.
    fn maxdf(&'a mut self, axis: UtahAxis) -> MaxIter<'a, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Max::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Max::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }
    }

    /// Get the minimum of entries along the specified `UtahAxis`.
    fn mindf(&'a mut self, axis: UtahAxis) -> MinIter<'a, f64, String> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Min::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Min::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }

    }



    /// Replace empty values with specified ImputeStrategy and along a `UtahAxis`.
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
