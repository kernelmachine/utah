# Utah

[![Build Status](https://travis-ci.org/pegasos1/rust-dataframe.svg?branch=master)](https://travis-ci.org/pegasos1/rust-dataframe)

**Utah** is a Rust crate for type-conscious tabular data manipulation with an expressive, functional interface. 

**Note**: This crate requires nightly for specialization to dataframes with `f64` data and `String` column/index labels. This will likely change in the future, as API stabilizes.

API currently in development and subject to change. 

For an in-depth introduction to the mechanics of this crate, as well as future goals, read this blog post: PLACEHOLDER

## Examples

#### Remove a row from a dataframe

```
let df: DataFrame<f64, String> = DataFrame::read_csv("test.csv").unwrap();       
let res = df.df_iter(UtahAxis::Row).remove(&["1"]).as_df();
```

#### Chain operations

```
let df: DataFrame<f64, String> = DataFrame::read_csv("test.csv").unwrap();       
let res = df.df_iter(UtahAxis::Row).remove(&["1"])
                                    .select(&["2"])
                                    .append("8", new_data.view())
                                    .sumdf()
                                    .as_df()?;
```

#### Create dataframes on the fly

```
let df = DataFrame<f64, String> = dataframe!(
    {
        "a" =>  column!([2., 3., 2.]),
        "b" =>  column!([2., NAN, 2.])
    });
```

#### Support mixed types 

```
let a = DataFrame<InnerType, OuterType> = dataframe!(
    {
        "name" =>  column!([InnerType::Str("Alice"), 
                            InnerType::Str("Bob"), 
                            InnerType::Str("Jane")]),
        "data" =>  column!([InnerType::Float(2.0), 
                              InnerType::Empty(), 
                              InnerType::Float(3.0)])
    });
let b: Result<DataFrame<InnerType, OuterType>> = DataFrame::read_csv("test.csv");
let res : DataFrame<InnerType, OuterType> = a.left_inner_join(&b).as_df();
```
