#[derive(Debug, Clone, PartialEq)]
pub struct MixedDataFrame<T, S, N, M, O, P>
    where T: Num,
          S: Identifier,
          M: ArrayLength<S> + Same<P>,
          P: ArrayLength<T> + Same<M>,
          N: ArrayLength<S> + Same<O>,
          O: ArrayLength<T> + Same<N>,
{
    pub columns: GenericArray<S, M>,
    pub data: Matrix<T>,
    pub index: GenericArray<S, N>,
    phantom_0: PhantomData<<N as Same<O>>::Output>,
    phantom_1: PhantomData<<P as Same<M>>::Output>,
}
