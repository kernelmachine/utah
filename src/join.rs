use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use types::*;
use std::iter::FromIterator;



pub struct InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    left: L,
    right: HashMap<OuterType, RowView<'a, InnerType>>,
}

pub struct OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    left: L,
    right: HashMap<OuterType, RowView<'a, InnerType>>,
}



impl<'a, L> InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new<RI>(left: L, right: RI) -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        InnerJoin {
            left: left,
            right: right.collect(),
        }
    }
}


impl<'a, L> OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new<RI>(left: L, right: RI) -> Self
        where RI: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {
        OuterJoin {
            left: left,
            right: right.collect(),
        }
    }
}


impl<'a, L> Iterator for InnerJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = (OuterType, RowView<'a, InnerType>, RowView<'a, InnerType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.left.next() {
                Some((k, lv)) => {
                    let rv = self.right.get(&k);
                    match rv {
                        Some(&v) => return Some((k, lv, v)),
                        None => continue,
                    }
                }
                None => return None,
            }

        }
    }
}



impl<'a, L> Iterator for OuterJoin<'a, L>
    where L: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    type Item = (OuterType, RowView<'a, InnerType>, Option<RowView<'a, InnerType>>);

    fn next(&mut self) -> Option<Self::Item> {

        match self.left.next() {
            Some((k, lv)) => {
                let rv = self.right.get(&k);
                match rv {
                    Some(&v) => return Some((k, lv, Some(v))),
                    None => Some((k, lv, None)),
                }

            }
            None => None,
        }

    }
}
