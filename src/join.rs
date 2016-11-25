use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;


pub enum JoinType {
    InnerJoin,
    OuterLeftJoin,
}
pub enum Join<L, K, RV> {
    InnerJoin { left: L, right: HashMap<K, RV> },
    OuterLeftJoin { left: L, right: HashMap<K, RV> },
}

impl<L, K, RV> Join<L, K, RV>
    where K: Hash + Eq
{
    pub fn new<LI, RI>(t: JoinType, left: LI, right: RI) -> Self
        where L: Iterator<Item = LI::Item>,
              LI: IntoIterator<IntoIter = L>,
              RI: IntoIterator<Item = (K, RV)>
    {
        match t {
            JoinType::InnerJoin => {
                Join::InnerJoin {
                    left: left.into_iter(),
                    right: right.into_iter().collect(),
                }
            }
            JoinType::OuterLeftJoin => {
                Join::OuterLeftJoin {
                    left: left.into_iter(),
                    right: right.into_iter().collect(),
                }
            }

        }

    }
}

impl<L, K, LV, RV> Iterator for Join<L, K, RV>
    where L: Iterator<Item = (K, LV)>,
          K: Hash + Eq + Debug,
          RV: Clone + Debug
{
    type Item = (K, LV, Option<RV>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            &mut Join::InnerJoin { ref mut left, ref right } => {
                loop {
                    match left.next() {
                        Some((k, lv)) => {
                            let rv = right.get(&k);
                            match rv {
                                Some(v) => return Some((k, lv, Some(v).cloned())),
                                None => continue,
                            }
                        }
                        None => return None,
                    }

                }
            }
            &mut Join::OuterLeftJoin { ref mut left, ref right } => {
                match left.next() {
                    Some((k, lv)) => {
                        let rv = right.get(&k);
                        Some((k, lv, rv.cloned()))
                    }
                    None => None,
                }
            }
        }

    }
}
