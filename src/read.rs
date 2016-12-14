// ///
// /// This example parses, sorts and groups the iris dataset
// /// and does some simple manipulations.
// ///
// /// Iterators and itertools functionality are used throughout.
// ///
// ///
//
// extern crate itertools;
//
// use itertools::Itertools;
// use std::collections::HashMap;
// use std::iter::repeat;
// use std::num::ParseFloatError;
// use std::str::FromStr;
// use types::{InnerType, Row};
// use ndarray::Array;
// use error::ErrorKind;
// // static DATA: &'static str = include_str!("iris.data");
//
// #[derive(Debug)]
// struct DataLine {
//     name: String,
//     data: Vec<InnerType>,
//     index: String,
// }
//
// #[derive(Clone, Debug)]
// enum ParseError {
//     Numeric(ParseFloatError),
// }
//
// impl From<ParseFloatError> for ParseError {
//     fn from(err: ParseFloatError) -> Self {
//         ParseError::Numeric(err)
//     }
// }
//
// /// Parse an Iris from a comma-separated line
// impl FromStr for DataLine {
//     type Err = ErrorKind;
//
//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         let mut dl = DataLine {
//             name: "".into(),
//             data: vec![],
//             index: "".into(),
//         };
//         let mut parts = s.split(",").map(str::trim);
//         // using Iterator::by_ref()
//         for (index, part) in parts.by_ref().enumerate() {
//             dl.data[index] = try!(part.parse::<InnerType>());
//         }
//
//
//
//         Ok(DataFrame::new(dl.data))
//     }
// }
