//! ## Table of contents
//!
//! + [DataFrame](#dataframe)
//! + [Combinators](#combinators)
//! + [Collection](#collection)
//! + [Mixed Types] (#mixed-types)
//!
//! ## DataFrame
//!
//! Utah is a dataframe crate backed by [ndarray](http://github.com/bluss/rust-ndarray) for type-conscious, tabular data manipulation with an expressive, functional interface.
//!
//! The dataframe allows users to access, transform, and compute over two-dimensional data that may or may not have mixed types.
//!
//!
//!
//! Please read [this](http://suchin.co/2016/12/27/Introducing-Utah) blog post for an in-depth introduction to the internals of this project.
//!
//! ### Creating a dataframe
//!
//! There are multiple ways to create a dataframe. The most straightforward way is to use a builder pattern:
//!
//! ```ignore
//! use utah::prelude::*;
//! let c = arr2(&[[2., 6.], [3., 4.], [2., 1.]]);
//! let mut df: DataFrame<f64> = DataFrame::new(c)
//!                                         .columns(&["a", "b"]).unwrap()
//!                                         .index(&["1", "2", "3"]).unwrap();
//! ```
//!
//! There's also a `dataframe!` macro which you can use to create new dataframes on the fly.
//!
//! Finally, you can import data from a CSV.
//!
//! ```ignore
//! use utah::prelude::*;
//! let file_name = "test.csv";
//! let df: Result<DataFrame<f64>> = DataFrame::read_csv(file_name);
//! ```
//!
//! Note that utah's `ReadCSV` trait is pretty barebones right now.
//!
//! ## Combinators
//!
//! The user interacts with Utah dataframes by chaining combinators, which are essentially iterator extensions (or _adapters_) over the original dataframe.
//!
//! The user interacts with Utah dataframes by chaining combinators, which are essentially iterator extensions (or _adapters_) over the original dataframe. This means that each operation is lazy by default. You can chain as many combinators as you want, but it won't do anything until you invoke a collection operation like `as_df`, which would allocate the results to a new dataframe, or `as_matrix`, which would allocate the results into an ndarray matrix.
//!
//! ### Transform combinators
//!
//! Transform combinators are meant for changing the shape of the data you're working with. Combinators in this class include `select`, `remove`, and `append`.
//!
//! ```ignore
//! use utah::prelude::*;
//! let a = arr2(&[[2, 7], [3, 4], [2, 8]]);
//! let df : DataFrame<i32> = DataFrame::new(a).index(&["1","2", "3"]).unwrap().columns(&["a", "b"]).unwrap();
//! let res = df.select(&["a", "c"], UtahAxis::Row);
//! ```
//!
//!
//! ### Process combinators
//!
//! Process combinators are meant for changing the original data you're working with. Combinators in this class include `impute` and `mapdf`. Impute replaces missing values of a dataframe with the mean of the corresponding column. Not that these operations require the use of a `MutableDataFrame`.
//!
//! ```ignore
//! use utah::prelude::*;
//! let mut a: MutableDataFrame<f64> = dataframe!(
//!     {
//!         "a" =>  column!([NAN, 3., 2.]),
//!         "b" =>  column!([2., NAN, 2.])
//!     });
//! let res = df.impute(ImputeStrategy::Mean, UtahAxis::Column);
//! ```
//!
//!
//! ### Interact combinators
//!
//! Interact combinators are meant for interactions between dataframes. They generally take at least two dataframe arguments. Combinators in this class include `inner_left_join`, `outer_left_join`, `inner_right_join`, `outer_right_join`, and `concat`.
//!
//! ```ignore
//! let a: DataFrame<f64> = dataframe!(
//!     {
//!         "a" =>  column!([NAN, 3., 2.]),
//!         "b" =>  column!([2., NAN, 2.])
//!     });
//! let b: DataFrame<f64> = dataframe!(
//!     {
//!         "b" =>  column!([NAN, 3., 2.]),
//!         "c" =>  column!([2., NAN, 2.])
//!     });
//! let res = a.inner_left_join(b).as_df()?;
//! ```
//!
//! ### Aggregate combinators
//!
//! Aggregate combinators are meant for reduction of a chain of combinators to some result. They are usually the last operation in a chain, but don't necessarily have to be. Combinators in this class include `sumdf`, `mindf`, `maxdf`, `stdev` (standard deviation), and `mean`. Currently, aggregate combinators are not iterator collection operations, because they do not invoke an iterator chain. This may change in the future.
//!
//! ```ignore
//! let a = arr2(&[[2.0, 7.0], [3.0, 4.0], [2.0, 8.0]]);
//! let df = DataFrame::new(a);
//! let res = df.mean(UtahAxis::Row);
//! ```
//!
//! ### Chaining combinators
//!
//! The real power in combinators come from the ability to chain them together in expressive transformations. I can do things like this:
//!
//! ```ignore
//! let result = df.df_iter(UtahAxis::Row)
//!         .remove(&["1"])
//!         .select(&["2"])
//!         .append("8", new_data.view())
//!         .inner_left_join(df_1)
//!         .sumdf()
//!         .as_df().unwrap();
//! ```
//!
//! Because we've built the chain on a row-wise dataframe iterator, each subsequent operation will only operate on the rows of the dataframe. If you want to operate on the columns, invoke a dataframe iterator with a `UtahAxis::Column`.

//! ### Collection
//!
//! There are many ways you can access or store the result of your chained operations. Because each data transformation is just an iterator, we can naturally collect the output of the chained operations via `collect()` or a `for loop`:
//!
//! ```ignore
//! for x in df.concat(&df_1) {
//!   println!("{:?}", x)
//! }
//! ```
//!
//! But we also have an `AsDataFrame` trait, which dumps the output of chained combinators into a new dataframe, matrix, or array, so we can do something like the following:
//!
//!
//! ```ignore
//! let maximum_values = df.concat(&df_1).maxdf(UtahAxis::Column).as_df()?;
//! ```
//!
//!
//! ### Mixed Types
//!
//! Now, I mentioned in the beginning that most dataframes provide mixed types, and I wanted to provide a similar functionality here. In the module `utah::mixtypes`, I've defined `InnerType`, which is an enum over various types of data that can coexist in the same dataframe:
//!
//! ```ignore
//! pub enum InnerType {
//!    Float(f64),
//!    Int64(i64),
//!    Int32(i32),
//!    Str(String),
//!    Empty,
//! }
//! ```
//!
//! I've also defined `OuterType`, which is an enum over the various types of *axis labels* that can coexist:
//!
//! ```ignore
//! pub enum OuterType {
//!    Str(String),
//!    Int64(i64),
//!    Int32(i32),
//!    USize(usize),
//! }
//! ```
//!
//!
//! With these wrappers, you can have Strings and f64s in the same dataframe.
//!
//! ```ignore
//! let file_name = "test.csv";
//! let df: Result<DataFrame<f64>> = DataFrame::read_csv(file_name);
//! ```
#![cfg_attr(nightly,test)]
#![cfg_attr(nightly,custom_derive)]
#![cfg_attr(nightly,stmt_expr_attributes)]
#![cfg_attr(nightly,specialization)]
#![recursion_limit = "1024"]

#[macro_use]


extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
#[cfg(nightly)]
extern crate test;
extern crate num;
#[macro_use]
extern crate error_chain;
extern crate itertools;
extern crate rustc_serialize;
extern crate csv;




pub mod combinators;
pub mod dataframe;
#[macro_use]
pub mod util;
mod implement;
pub mod mixedtypes;
mod bench;
#[macro_use]
mod tests;

pub mod prelude;
