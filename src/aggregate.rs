
use types::*;


#[derive(Clone)]
pub struct Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Sum<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Sum<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Sum { data: df }
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Mean<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Mean<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Mean { data: df }
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Max<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Max<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Max { data: df }
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Min<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Min<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Min { data: df }
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
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    data: I,
}

impl<'a, I> Stdev<'a, I>
    where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
{
    pub fn new(df: I) -> Stdev<'a, I>
        where I: Iterator<Item = (OuterType, RowView<'a, InnerType>)>
    {

        Stdev { data: df }
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
