use std::collections::HashMap;
use typenum::*;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub struct GenericHashMap<A: Hash + Eq, B, N: Unsigned> {
    data: HashMap<A, B>,
    length: N,
}

impl<A, B, N> GenericHashMap<A, B, N>
    where A: Hash + Eq,
          N: Unsigned + Sized
{
    fn new(n: N) -> GenericHashMap<A, B, N> {
        GenericHashMap {
            data: HashMap::with_capacity(N::to_usize()),
            length: n,
        }
    }
}

impl Iterator for GenericHashMap<A, B, N> {}
