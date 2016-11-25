use std::collections::BTreeMap;
use dataframe::*;


pub fn merge_maps(first_context: &BTreeMap<IndexType, usize>,
                  second_context: &BTreeMap<IndexType, usize>)
                  -> BTreeMap<IndexType, usize> {
    let mut new_context: BTreeMap<IndexType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    for (key, value) in second_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }
    new_context
}

pub fn concat_index_maps(first_context: &BTreeMap<IndexType, usize>,
                         second_context: &BTreeMap<IndexType, usize>)
                         -> BTreeMap<IndexType, usize> {
    let mut new_context: BTreeMap<IndexType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            IndexType::Str(z) => {
                new_context.insert(IndexType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            IndexType::Date(z) => {
                new_context.insert(IndexType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }

        };
    }
    new_context
}

pub fn concat_column_maps(first_context: &BTreeMap<ColumnType, usize>,
                          second_context: &BTreeMap<ColumnType, usize>)
                          -> BTreeMap<ColumnType, usize> {
    let mut new_context: BTreeMap<ColumnType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            ColumnType::Str(z) => {
                new_context.insert(ColumnType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            ColumnType::Date(z) => {
                new_context.insert(ColumnType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
        };
    }
    new_context
}
