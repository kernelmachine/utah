
use types::*;
use std::iter::{Iterator, Chain};
use aggregate::*;
use transform::*;
use process::*;
use dataframe::{DataFrame, MutableDataFrame};
use ndarray::Array;

pub trait Aggregate<'a> {
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;
}
pub trait Process<'a> {
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone;
    fn to_mut_df(self) -> MutableDataFrame<'a>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone;
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone;
}
pub trait Transform<'a> {
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>,
              T: 'a;
    fn concat<I>(self, other: I) -> Chain<Self, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>;

    fn inner_left_join<I>(self, other: I) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone;

    fn outer_left_join<I>(self, other: I) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone;

    fn inner_right_join<I>(self, other: I) -> InnerJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone;

    fn outer_right_join<I>(self, other: I) -> OuterJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone;

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              F: Fn(&InnerType) -> B;
}


impl<'a> Aggregate<'a> for DataFrameIterator<'a> {
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}


impl<'a> Transform<'a> for DataFrameIterator<'a> {
    fn to_df(self) -> DataFrame {
        let axis = self.axis;
        let other = self.other;
        let data = self.data;
        let names = self.names;
        let mut c = Vec::new();
        let mut n = Vec::new();

        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (other.len(), 0),
            UtahAxis::Column => (0, other.len()),
        };

        for (i, j) in names.zip(data) {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }


        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }


    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names: Vec<OuterType> = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }

    fn concat<I>(self, other: I) -> Chain<Self, I>
        where I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }



    fn inner_left_join<I>(self, other: I) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }

    fn outer_left_join<I>(self, other: I) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }

    fn inner_right_join<I>(self, other: I) -> InnerJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        InnerJoin::new(other, self)
    }

    fn outer_right_join<I>(self, other: I) -> OuterJoin<'a, I>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              I: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}




impl<'a, I> Aggregate<'a> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}

impl<'a, I> Transform<'a> for Select<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (s.other.len(), 0),
            UtahAxis::Column => (0, s.other.len()),
        };

        for (i, j) in s {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }



    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other.clone(), axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other.clone(), axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>,
              T: 'a
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}





impl<'a, I> Aggregate<'a> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}

impl<'a, I> Transform<'a> for Remove<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (s.other.len(), 0),
            UtahAxis::Column => (0, s.other.len()),
        };

        for (i, j) in s {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}



impl<'a, I> Aggregate<'a> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn sumdf(self) -> Sum<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Sum::new(self)
    }

    fn max(self) -> Max<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Max::new(self)
    }

    fn min(self) -> Min<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Min::new(self)
    }

    fn mean(self) -> Mean<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Mean::new(self)
    }

    fn stdev(self) -> Stdev<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        Stdev::new(self)
    }
}
impl<'a, I> Transform<'a> for Append<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let axis = self.axis;
        let other = self.other;
        let data = self.new_data;
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (other.len(), 0),
            UtahAxis::Column => (0, other.len()),
        };

        for (i, j) in data {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }


        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: n,
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }

    fn select<T>(self, names: &'a [T]) -> Select<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Select::new(self, names, other, axis)
    }


    fn remove<T>(self, names: &'a [T]) -> Remove<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let names = names.iter()
            .map(|x| OuterType::from(x))
            .collect();
        Remove::new(self, names, other, axis)

    }

    fn append<T>(self, name: &'a T, data: RowView<'a, InnerType>) -> Append<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              OuterType: From<&'a T>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let name = OuterType::from(name);
        Append::new(self, name, data, other, axis)

    }
    fn concat<J>(self, other: J) -> Chain<Self, J>
        where J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        self.chain(other)
    }

    fn inner_left_join<J>(self, other: J) -> InnerJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin::new(self, other)
    }
    fn outer_left_join<J>(self, other: J) -> OuterJoin<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin::new(self, other)
    }
    fn inner_right_join<J>(self, other: J) -> InnerJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        InnerJoin::new(other, self)
    }
    fn outer_right_join<J>(self, other: J) -> OuterJoin<'a, J>
        where Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone,
              J: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        OuterJoin::new(other, self)
    }

    fn mapdf<F, B>(self, f: F) -> MapDF<'a, Self, F, B>
        where F: Fn(&InnerType) -> B,
              Self: Sized + Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
    {
        let axis = self.axis.clone();
        MapDF::new(self, f, axis)
    }
}

impl<'a, I> Process<'a> for Impute<'a, I>
    where I: Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
{
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {
        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone
    {

        let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (s.other.len(), 0),
            UtahAxis::Column => (0, s.other.len()),
        };

        for (i, j) in s {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                MutableDataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: n,
                }
            }
            UtahAxis::Column => {
                MutableDataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: other,
                }
            }

        }

    }
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone
    {
        self.to_mut_df().to_df()
    }
}

impl<'a> Process<'a> for MutableDataFrameIterator<'a> {
    fn impute(self, strategy: ImputeStrategy) -> Impute<'a, Self>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)>
    {

        let other = self.other.clone();
        let axis = self.axis.clone();
        Impute::new(self, strategy, other, axis)
    }

    fn to_mut_df(self) -> MutableDataFrame<'a>
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone
    {
        let s = self.clone();
        let axis = self.axis.clone();
        let other = self.other.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let (mut ncols, mut nrows) = match axis {
            UtahAxis::Row => (s.other.len(), 0),
            UtahAxis::Column => (0, s.other.len()),
        };

        for (i, j) in s {
            match axis {
                UtahAxis::Row => nrows += 1,
                UtahAxis::Column => ncols += 1,
            };

            c.extend(j);
            n.push(i.to_owned());
        }



        match axis {
            UtahAxis::Row => {
                MutableDataFrame {
                    columns: other,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: n,
                }
            }
            UtahAxis::Column => {
                MutableDataFrame {
                    columns: n,
                    data: Array::from_shape_vec((nrows, ncols), c).unwrap(),
                    index: other,
                }
            }

        }
    }
    fn to_df(self) -> DataFrame
        where Self: Sized + Iterator<Item = (OuterType, RowViewMut<'a, InnerType>)> + Clone
    {
        self.to_mut_df().to_df()
    }
}
