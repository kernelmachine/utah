
macro_rules! dataframe {
    ($name : ident,($($field: ident),*), ($($value: expr),*)) => {
        #[derive(Debug, PartialEq)]
        struct $name { $($field: Vec<i32>),* };
        impl Default for $name {
            fn default() -> $name {
                $name {
                $($field : $value),*
                }
            }
        }

    }
}





#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        dataframe!(MyData,
                   (a,b),
                   (vec![1,2,3], vec![4,5,6]));
        assert_eq!(MyData {a : vec![1,2,3], b : vec![4,5,6]},MyData::default())
    }
}
