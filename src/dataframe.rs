use ndarray::Axis;
use error::*;
use types::*;
use std::string::ToString;
use std::iter::Iterator;
use aggregate::*;
use transform::*;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    pub columns: Vec<OuterType>,
    pub data: Matrix<InnerType>,
    pub index: Vec<OuterType>,
}



impl DataFrame {
    pub fn new<T: Clone>(data: Matrix<T>) -> DataFrame
        where InnerType: From<T>
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


    pub fn columns<'a, T>(mut self, columns: &'a [T]) -> Result<DataFrame>
        where OuterType: From<&'a T>
    {
        if columns.len() != self.data.shape()[1] {
            return Err(ErrorKind::ColumnShapeMismatch.into());
        }
        self.columns = columns.iter()
            .map(|x| OuterType::from(x))
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
            .collect();
        Ok(self)
    }

    pub fn df_iter<'a>(&'a self, axis: Axis) -> DataFrameIterator<'a> {
        match axis {
            Axis(0) => {
                DataFrameIterator {
                    names: self.index.iter(),
                    data: self.data.axis_iter(Axis(0)),
                }
            }
            Axis(1) => {
                DataFrameIterator {
                    names: self.columns.iter(),
                    data: self.data.axis_iter(Axis(1)),
                }
            }
            _ => panic!(),

        }
    }
    pub fn select<'a, T>(&'a self, names: &'a [T], axis: Axis) -> Select<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            Axis(0) => Select::new(self.df_iter(Axis(0)), names),
            Axis(1) => Select::new(self.df_iter(Axis(1)), names),
            _ => panic!(),

        }
    }


    pub fn remove<'a, T>(&'a self, names: &'a [T], axis: Axis) -> Remove<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        match axis {
            Axis(0) => Remove::new(self.df_iter(Axis(0)), names),
            Axis(1) => Remove::new(self.df_iter(Axis(1)), names),
            _ => panic!(),

        }
    }

    pub fn append<'a, T>(&'a self,
                         name: &'a T,
                         data: RowView<'a, InnerType>,
                         axis: Axis)
                         -> Append<'a, DataFrameIterator<'a>>
        where OuterType: From<&'a T>
    {
        let name = OuterType::from(name);
        match axis {
            Axis(0) => Append::new(self.df_iter(Axis(0)), name, data),
            Axis(1) => Append::new(self.df_iter(Axis(1)), name, data),
            _ => panic!(),

        }
    }

    pub fn inner_left_join<'a>(&'a self,
                               other: &'a DataFrame,
                               axis: Axis)
                               -> InnerJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => InnerJoin::new(self.df_iter(Axis(0)), other.df_iter(Axis(0))),
            Axis(1) => InnerJoin::new(self.df_iter(Axis(1)), other.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
    pub fn outer_left_join<'a>(&'a self,
                               other: &'a DataFrame,
                               axis: Axis)
                               -> OuterJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => OuterJoin::new(self.df_iter(Axis(0)), other.df_iter(Axis(0))),
            Axis(1) => OuterJoin::new(self.df_iter(Axis(1)), other.df_iter(Axis(1))),
            _ => panic!(),

        }

    }
    pub fn inner_right_join<'a>(&'a self,
                                other: &'a DataFrame,
                                axis: Axis)
                                -> InnerJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => InnerJoin::new(other.df_iter(Axis(0)), self.df_iter(Axis(0))),
            Axis(1) => InnerJoin::new(other.df_iter(Axis(1)), self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }
    pub fn outer_right_join<'a>(&'a self,
                                other: &'a DataFrame,
                                axis: Axis)
                                -> OuterJoin<'a, DataFrameIterator<'a>> {
        match axis {
            Axis(0) => OuterJoin::new(other.df_iter(Axis(0)), self.df_iter(Axis(0))),
            Axis(1) => OuterJoin::new(other.df_iter(Axis(1)), self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }



    pub fn sumdf<'a>(&'a self, axis: Axis) -> Sum<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Sum::new(self.df_iter(Axis(0))),
            Axis(1) => Sum::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn map<'a, F, B>(&'a self, f: F, axis: Axis) -> MapDF<'a, DataFrameIterator<'a>, F, B>
        where F: Fn(&InnerType) -> B
    {

        match axis {
            Axis(0) => MapDF::new(self.df_iter(Axis(0)), f),
            Axis(1) => MapDF::new(self.df_iter(Axis(1)), f),
            _ => panic!(),

        }
    }

    pub fn mean<'a>(&'a self, axis: Axis) -> Mean<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Mean::new(self.df_iter(Axis(0))),
            Axis(1) => Mean::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn max<'a>(&'a self, axis: Axis) -> Max<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Max::new(self.df_iter(Axis(0))),
            Axis(1) => Max::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn min<'a>(&'a self, axis: Axis) -> Min<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Min::new(self.df_iter(Axis(0))),
            Axis(1) => Min::new(self.df_iter(Axis(1))),
            _ => panic!(),

        }
    }

    pub fn stdev<'a>(&'a self, axis: Axis) -> Stdev<'a, DataFrameIterator<'a>> {

        match axis {
            Axis(0) => Stdev::new(self.df_iter(Axis(0))),
            Axis(1) => Stdev::new(self.df_iter(Axis(1))),
            _ => panic!(),

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
