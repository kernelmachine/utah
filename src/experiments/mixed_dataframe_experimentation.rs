#[derive(Clone)]
pub struct MixedDataFrameRowIterator<'a, T: 'a, S: 'a>
    where T: Num,
          S: Identifier,
          Zip<AxisIter<'a, T, usize>>: Iterator
{
    pub names: Iter<'a, S>,
    pub data: Zip<AxisIter<'a, T, usize>>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}

#[derive(Clone)]
pub struct MixedDataFrameColIterator<'a, T: 'a, S: 'a>
    where T: Num,
          S: Identifier,
          BTreeIter<'a, S, Row<T>>: Iterator
{
    pub names: Iter<'a, S>,
    pub data: BTreeIter<'a, S, Row<T>>,
    pub other: Vec<S>,
    pub axis: UtahAxis,
}


impl<'a, T, S> Iterator for MixedDataFrameRowIterator<'a, T, S>
    where T: Num,
          S: Identifier,
          Zip<AxisIter<'a, T, usize>>: Iterator
{
    type Item = (S, <Zip<AxisIter<'a, T, usize>> as Iterator>::Item);
    fn next(&mut self) -> Option<Self::Item> {

        match self.names.next() {
            Some(val) => {
                match self.data.next() {
                    Some(dat) => Some((val.clone(), dat)),
                    None => None,
                }
            }
            None => None,
        }
    }
}

impl<'a, T, S> Iterator for MixedDataFrameColIterator<'a, T, S>
    where T: Num,
          S: Identifier,
          BTreeIter<'a, S, Row<T>>: Iterator
{
    type Item = <BTreeIter<'a, S, Row<T>> as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.data.next()
    }
}
