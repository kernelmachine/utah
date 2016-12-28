use util::types::*;
use std::iter::Iterator;
use combinators::aggregate::*;
use combinators::process::*;
use combinators::interact::*;
use combinators::transform::*;
use util::types::UtahAxis;
use util::traits::*;
use dataframe::*;

impl<'a, T, S> Operations<'a, T, S> for DataFrame<T, S>
    where T: 'a + Num,
          S: Identifier
{
    /// Get the dimensions of the dataframe.
    default fn shape(self) -> (usize, usize) {
        self.data.dim()
    }



    /// Select rows or columns over the specified `UtahAxis`.
    default fn select<U: ?Sized>(&'a self,
                                 names: &'a [&'a U],
                                 axis: UtahAxis)
                                 -> SelectIter<'a, T, S>
        where S: From<&'a U>
    {
        let names: Vec<S> = names.iter()
            .map(|x| (*x).into())
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
    default fn remove<U: ?Sized>(&'a self,
                                 names: &'a [&'a U],
                                 axis: UtahAxis)
                                 -> RemoveIter<'a, T, S>
        where S: From<&'a U>
    {
        let names: Vec<S> = names.iter()
            .map(|x| (*x).into())
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
    default fn append<U: ?Sized>(&'a mut self,
                                 name: &'a U,
                                 data: RowView<'a, T>,
                                 axis: UtahAxis)
                                 -> AppendIter<'a, T, S>
        where S: From<&'a U>
    {
        let name: S = name.into();
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
    default fn inner_left_join(&'a self, other: &'a DataFrame<T, S>) -> InnerJoinIter<'a, T, S> {
        InnerJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())
    }

    /// Perform an outer left join between two dataframes along the specified `UtahAxis`.
    default fn outer_left_join(&'a self, other: &'a DataFrame<T, S>) -> OuterJoinIter<'a, T, S> {

        OuterJoin::new(self.df_iter(UtahAxis::Row),
                       other.df_iter(UtahAxis::Row),
                       self.columns.clone(),
                       other.columns.clone())
    }

    /// Perform an inner right join between two dataframes along the specified `UtahAxis`.
    default fn inner_right_join(&'a self, other: &'a DataFrame<T, S>) -> InnerJoinIter<'a, T, S> {
        InnerJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    /// Perform an outer right join between two dataframes along the specified `UtahAxis`.
    default fn outer_right_join(&'a self, other: &'a DataFrame<T, S>) -> OuterJoinIter<'a, T, S> {
        OuterJoin::new(other.df_iter(UtahAxis::Row),
                       self.df_iter(UtahAxis::Row),
                       other.columns.clone(),
                       self.columns.clone())

    }

    default fn concat(&'a self,
                      other: &'a DataFrame<T, S>,
                      axis: UtahAxis)
                      -> ConcatIter<'a, T, S> {
        match axis {
            UtahAxis::Row => {
                Concat::new(self.df_iter(UtahAxis::Column),
                            other.df_iter(UtahAxis::Column),
                            self.columns.clone(),
                            UtahAxis::Column)
            }
            UtahAxis::Column => {
                Concat::new(self.df_iter(UtahAxis::Row),
                            other.df_iter(UtahAxis::Row),
                            self.columns.clone(),
                            UtahAxis::Row)
            }
        }
    }


    /// Sum along the specified `UtahAxis`.
    default fn sumdf(&'a mut self, axis: UtahAxis) -> SumIter<'a, T, S> {
        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Sum::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Sum::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }
    }

    /// Map a function along the specified `UtahAxis`.
    default fn map<F>(&'a mut self, f: F, axis: UtahAxis) -> MapDFIter<'a, T, S, F>
        where F: Fn(T) -> T
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
    default fn mean(&'a mut self, axis: UtahAxis) -> MeanIter<'a, T, S> {

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
    default fn maxdf(&'a mut self, axis: UtahAxis) -> MaxIter<'a, T, S> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Max::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Max::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }
    }

    /// Get the minimum of entries along the specified `UtahAxis`.
    default fn mindf(&'a mut self, axis: UtahAxis) -> MinIter<'a, T, S> {

        let columns = self.columns.clone();
        let index = self.index.clone();
        match axis {
            UtahAxis::Row => Min::new(self.df_iter(UtahAxis::Row), index, UtahAxis::Row),
            UtahAxis::Column => Min::new(self.df_iter(UtahAxis::Column), columns, UtahAxis::Column),

        }

    }

    /// Replace empty values with specified ImputeStrategy along the specified `UtahAxis`.
    default fn impute(&'a mut self,
                      strategy: ImputeStrategy,
                      axis: UtahAxis)
                      -> ImputeIter<'a, T, S> {

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
