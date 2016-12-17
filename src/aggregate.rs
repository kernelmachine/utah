
use types::*;
use traits::*;
use dataframe::*;
use ndarray::Array;

#[derive(Clone)]
pub struct Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + 'a
{
    data: I,
    other: Vec<OuterType>,
    axis: UtahAxis,
}

impl<'a, I> Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: Vec<OuterType>, axis: UtahAxis) -> Sum<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Sum {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I> Iterator for Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => {
                return Some((0..dat.len()).fold(dat.get(0).unwrap().to_owned(),
                                                |x, y| x + dat.get(y).unwrap().to_owned()))
            }
        }
    }
}

#[derive(Clone)]
pub struct Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + 'a
{
    data: I,
    other: Vec<OuterType>,
    axis: UtahAxis,
}

impl<'a, I> Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: Vec<OuterType>, axis: UtahAxis) -> Mean<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Mean {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I> Iterator for Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                match dat.uget(0) {
                    &InnerType::Float(_) => return Some(sum / InnerType::Float(size as f64)),
                    &InnerType::Int32(_) => return Some(sum / InnerType::Int32(size as i32)),
                    &InnerType::Int64(_) => return Some(sum / InnerType::Int64(size as i64)),
                    _ => return Some(InnerType::Empty),
                }

            },
        }
    }
}


#[derive(Clone)]
pub struct Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + 'a
{
    data: I,
    other: Vec<OuterType>,
    axis: UtahAxis,
}

impl<'a, I> Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: Vec<OuterType>, axis: UtahAxis) -> Max<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Max {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I> Iterator for Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().max().map(|x| x.to_owned()),
        }



    }
}


#[derive(Clone)]
pub struct Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + 'a
{
    data: I,
    other: Vec<OuterType>,
    axis: UtahAxis,
}

impl<'a, I> Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: Vec<OuterType>, axis: UtahAxis) -> Min<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Min {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I> Iterator for Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            None => return None,
            Some((_, dat)) => return dat.iter().min().map(|x| x.to_owned()),
        }



    }
}

#[derive(Clone)]
pub struct Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + 'a
{
    data: I,
    other: Vec<OuterType>,
    axis: UtahAxis,
}

impl<'a, I> Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I, other: Vec<OuterType>, axis: UtahAxis) -> Stdev<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Stdev {
            data: df,
            other: other,
            axis: axis,
        }
    }
}

impl<'a, I> Iterator for Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = InnerType;
    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {

            None => return None,
            Some((_, dat)) => unsafe {
                let size = dat.len();
                let first_element = dat.uget(0).to_owned();
                let sum = (0..size).fold(first_element, |x, y| x + dat.uget(y).to_owned());

                let mean = match dat.uget(0) {
                    &InnerType::Float(_) => sum / InnerType::Float(size as f64),
                    &InnerType::Int32(_) => sum / InnerType::Int32(size as i32),
                    &InnerType::Int64(_) => sum / InnerType::Int64(size as i64),
                    _ => InnerType::Empty,
                };

                let stdev = (0..size).fold(dat.uget(0).to_owned(), |x, y| {
                    x +
                    (dat.uget(y).to_owned() - mean.to_owned()) *
                    (dat.uget(y).to_owned() - mean.to_owned())
                });


                Some(stdev)


            },
        }



    }
}


impl<'a, I> ToDataFrame<'a, InnerType> for Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType> for Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}



impl<'a, I> ToDataFrame<'a, InnerType> for Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType> for Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}


impl<'a, I> ToDataFrame<'a, InnerType> for Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {
        let other = self.other.clone();
        let axis = self.axis.clone();
        let c: Vec<_> = self.collect();
        let res_dim = match axis {
            UtahAxis::Row => (1, other.len()),
            UtahAxis::Column => (other.len(), 1),
        };



        match axis {
            UtahAxis::Row => {
                DataFrame {
                    columns: other,
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: vec![OuterType::Int32(1)],
                }
            }
            UtahAxis::Column => {
                DataFrame {
                    columns: vec![OuterType::Int32(1)],
                    data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
                    index: other,
                }
            }

        }
    }
}
