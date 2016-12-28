# Utah

[![Build Status](https://travis-ci.org/pegasos1/utah.svg?branch=master)](https://travis-ci.org/pegasos1/utah)

[**Utah**](http://crates.io/crates/utah) is a Rust crate backed by [ndarray](https://github.com/bluss/rust-ndarray) for type-conscious, tabular data manipulation with an expressive, functional interface.

**Note**: This crate works on stable. However, if you are working with dataframes with `f64` data, use nightly, because you will get performance benefits of specialization. 

API currently in development and subject to change.

For an in-depth introduction to the mechanics of this crate, as well as future goals, read [this](http://suchin.co/2016/28/12/Introducing-Utah) blog post.

## Install

Add the following to your `Cargo.toml`:

```
utah="0.1.2"
```

And add the following to your `lib.rs` or `main.rs`

```
#[macro_use]
extern crate utah
```
## Documentation

Check out [docs.rs](http://docs.rs/utah) for latest documentation. 

## Examples


#### Create dataframes on the fly

```rust
use utah::prelude::*;
let df = DataFrame<f64> = dataframe!(
    {
        "a" =>  column!([2., 3., 2.]),
        "b" =>  column!([2., NAN, 2.])
    });

let a = arr2(&[[2.0, 7.0], [3.0, 4.0]]);
let df : Result<DataFrame<f64>> = DataFrame::new(a).index(&["1", "2"]);
```

#### Transform the dataframe

```rust
use utah::prelude::*;
let df: DataFrame<f64> = DataFrame::read_csv("test.csv")?;       
let res : DataFrame<f64> = df.remove(&["a", "c"], UtahAxis::Column).as_df()?;
```

#### Chain operations

```rust
use utah::prelude::*;
let df: DataFrame<f64> = DataFrame::read_csv("test.csv").unwrap();       
let res : DataFrame<f64> = df.df_iter(UtahAxis::Row)
                                     .remove(&["1"])
                                     .select(&["2"])
                                     .append("8", new_data.view())
                                     .sumdf()
                                     .as_df()?;
```

#### Support mixed types

```rust
use utah::prelude::*;
let a = DataFrame<InnerType> = dataframe!(
    {
        "name" =>  column!([InnerType::Str("Alice"),
                            InnerType::Str("Bob"),
                            InnerType::Str("Jane")]),
        "data" =>  column!([InnerType::Float(2.0),
                            InnerType::Empty(),
                            InnerType::Float(3.0)])
    });
let b: DataFrame<InnerType> = DataFrame::read_csv("test.csv")?;
let res : DataFrame<InnerType> = a.left_inner_join(&b).as_df()?;
```
