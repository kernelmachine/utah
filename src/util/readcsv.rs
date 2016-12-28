///
/// This example parses, sorts and groups the iris dataset
/// and does some simple manipulations.
///
/// Iterators and itertools functionality are used throughout.
///
///

use ndarray::Array;
use dataframe::DataFrame;
use util::traits::{Num, Identifier};
use util::error::*;
use util::traits::Constructor;
use rustc_serialize::Decodable;

use csv;



pub trait ReadCSV<T, S>
    where T: Num + Decodable,
          S: Identifier
{
    fn read_csv(file: &'static str) -> Result<DataFrame<T>>;
}

impl<T, S> ReadCSV<T, S> for DataFrame<T>
    where T: Num + Decodable,
          S: Identifier + From<String>
{
    fn read_csv(file: &'static str) -> Result<DataFrame<T>> {
        let mut rdr = csv::Reader::from_file(file).unwrap();
        let columns = rdr.headers().unwrap();
        let (mut nrow, ncol) = (0, columns.len());
        let mut v: Vec<T> = Vec::new();
        for record in rdr.decode() {
            nrow += 1;
            let e: Vec<T> = record.unwrap();
            v.extend(e.into_iter())
        }

        let matrix = Array::from_shape_vec((nrow, ncol), v).unwrap();
        DataFrame::new(matrix).columns(&columns[..])
    }
}
