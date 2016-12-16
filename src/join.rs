use traits::Join;
use types::*;
use std::iter::Iterator;
use std::iter::repeat;
use std::collections::HashMap;
use ndarray::Array;

use dataframe::*;

#[derive(Clone)]
pub struct InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub left: L,
    pub right: HashMap<OuterType, RowView<'a, InnerType>>,
    pub left_columns: Vec<OuterType>,
    pub right_columns: Vec<OuterType>,
}

impl<'a, L> InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub fn new<RI>(left: L,
                   right: RI,
                   left_columns: Vec<OuterType>,
                   right_columns: Vec<OuterType>)
                   -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin {
            left: left,
            right: right.collect(),
            left_columns: left_columns,
            right_columns: right_columns,
        }
    }
}



impl<'a, L> Iterator for InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    type Item = (OuterType, RowView<'a, InnerType>, RowView<'a, InnerType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.next() {
                Some((k, lv)) => {
                    let rv = self.right.get(&k);
                    match rv {
                        Some(v) => return Some((k, lv, *v)),
                        None => continue,
                    }
                }
                None => return None,
            }

        }
    }
}

#[derive(Clone)]
pub struct OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    left: L,
    right: HashMap<OuterType, RowView<'a, InnerType>>,
    left_columns: Vec<OuterType>,
    right_columns: Vec<OuterType>,
}


impl<'a, L> OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    pub fn new<RI>(left: L,
                   right: RI,
                   left_columns: Vec<OuterType>,
                   right_columns: Vec<OuterType>)
                   -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin {
            left: left,
            right: right.collect(),
            left_columns: left_columns,
            right_columns: right_columns,
        }
    }
}


impl<'a, L> Iterator for OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    type Item = (OuterType, Row<InnerType>, Option<Row<InnerType>>);

    fn next(&mut self) -> Option<Self::Item> {

        match self.left.next() {
            Some((k, lv)) => {
                let rv = self.right.get(&k);
                match rv {
                    Some(v) => return Some((k, lv.to_owned(), Some(v.to_owned()))),
                    None => Some((k, lv.to_owned(), None)),
                }

            }
            None => None,
        }

    }
}


impl<'a, L> Join<'a> for InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {

        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), right_columns.len() + left_columns.len());


        for (i, j, k) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            c.extend(k.iter().map(|x| x.to_owned()));

            n.push(i.to_owned());
        }

        let columns: Vec<_> = right_columns.iter()
            .chain(left_columns.iter())
            .map(|x| x.to_owned())
            .collect();

        DataFrame {
            columns: columns,
            data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
            index: n,
        }


    }
}


impl<'a, L> Join<'a> for OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)> + Clone
{
    fn to_df(self) -> DataFrame {

        let s = self.clone();
        let right_columns = self.right_columns.clone();
        let left_columns = self.left_columns.clone();
        let mut c = Vec::new();
        let mut n = Vec::new();
        let res_dim = (s.fold(0, |acc, _| acc + 1), right_columns.len() + left_columns.len());


        for (i, j, k) in self {
            c.extend(j.iter().map(|x| x.to_owned()));
            match k {
                Some(z) => c.extend(z.iter().map(|x| x.to_owned())),
                None => c.extend(repeat(InnerType::Empty).take(right_columns.len())),
            }


            n.push(i.to_owned());
        }

        let columns: Vec<_> = right_columns.iter()
            .chain(left_columns.iter())
            .map(|x| x.to_owned())
            .collect();

        DataFrame {
            columns: columns,
            data: Array::from_shape_vec(res_dim, c).unwrap().mapv(|x| x.to_owned()),
            index: n,
        }


    }
}
