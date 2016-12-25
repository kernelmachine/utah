
use util::types::*;
use util::traits::*;
use dataframe::*;
use ndarray::Array;
use util::error::*;

#[derive(Clone, Debug)]
pub struct Sum<'a, I: 'a, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Num,
          S: Identifier
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Sum<'a, I, T, S> {

        Sum {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => return Some(dat.scalar_sum()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mean<'a, I: 'a, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + 'a,
          S: Identifier
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Mean<'a, I, T, S> {

        Mean {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + 'a,
          S: Identifier
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            Some((_, dat)) => {
                let size = dat.fold(T::one(), |acc, _| acc + T::one());
                let mean = dat.scalar_sum() / size;
                Some(mean)
            }
            None => return None,

        }
    }
}


#[derive(Clone)]
pub struct Max<'a, I: 'a, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + 'a,
          S: Identifier
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Max<'a, I, T, S> {

        Max {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + Ord + 'a,
          S: Identifier
{
    type Item = T;
    default fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().max().map(|x| x.clone()),
        }



    }
}


#[derive(Clone, Debug)]
pub struct Min<'a, I: 'a, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + 'a,
          S: Identifier
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Min<'a, I, T, S> {

        Min {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + Ord,
          S: Identifier
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().min().map(|x| x.clone()),
        }



    }
}

#[derive(Clone)]
pub struct Stdev<'a, I: 'a, T: 'a, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    data: I,
    other: Vec<S>,
    axis: UtahAxis,
}

impl<'a, I, T, S> Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + 'a,
          S: Identifier
{
    pub fn new(df: I, other: Vec<S>, axis: UtahAxis) -> Stdev<'a, I, T, S> {

        Stdev {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I, T, S> Iterator for Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.fold(T::one(), |acc, _| acc + T::one());
                let mean = dat.scalar_sum() / size;

                let stdev = (0..dat.len()).fold(dat.uget(0).to_owned(), |x, y| {
                    x +
                    (dat.uget(y).to_owned() - mean.to_owned()) *
                    (dat.uget(y).to_owned() - mean.to_owned())
                });


                Some(stdev)


            },
        }



    }
}


impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Stdev<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)> + 'a,
          T: Num + 'a,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };


        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec![S::default()];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}



impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Mean<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec![S::default()];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}



impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Max<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + Ord,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec![S::default()];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }

        }

    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }
    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}


impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Min<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num + Ord,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec![S::default()];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}


impl<'a, I, T, S> ToDataFrame<'a, T, T, S> for Sum<'a, I, T, S>
    where I: Iterator<Item = (S, RowView<'a, T>)>,
          T: Num,
          S: Identifier
{
    fn as_df(self) -> Result<DataFrame<T, S>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        let d = Array::from_shape_vec(res_dim, c).unwrap();
        let def = vec![S::default()];
        match axis {
            UtahAxis::Row => {
                let df = DataFrame::new(d).columns(&other[..])?.index(&def[..])?;
                Ok(df)
            }
            UtahAxis::Column => {
                let df = DataFrame::new(d).columns(&def[..])?.index(&other[..])?;
                Ok(df)
            }

        }
    }
    fn as_matrix(self) -> Result<Matrix<T>> {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };

        Ok(Array::from_shape_vec(res_dim, c).unwrap())


    }

    fn as_array(self) -> Result<Row<T>> {

        let c: Vec<_> = self.collect();
        Ok(Array::from_vec(c))
    }
}
