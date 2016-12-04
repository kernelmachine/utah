use std::collections::BTreeMap;
use types::*;


pub fn merge_maps(first_context: &BTreeMap<OuterType, usize>,
                  second_context: &BTreeMap<OuterType, usize>)
                  -> BTreeMap<OuterType, usize> {
    let mut new_context: BTreeMap<OuterType, usize> = BTreeMap::new();

    for (key, value) in first_context.iter() {
        if second_context.contains_key(key) {
            new_context.insert(key.to_owned(), *value);
        }
    }
    for (key, value) in second_context.iter() {
        if first_context.contains_key(key) {
            new_context.insert(key.to_owned(), *value);
        }
    }
    new_context
}

pub fn concat_index_maps(first_context: &BTreeMap<OuterType, usize>,
                         second_context: &BTreeMap<OuterType, usize>)
                         -> BTreeMap<OuterType, usize> {
    let mut new_context: BTreeMap<OuterType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            OuterType::Str(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            OuterType::Date(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            OuterType::Int(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }

        };
    }
    new_context
}

pub fn concat_column_maps(first_context: &BTreeMap<OuterType, usize>,
                          second_context: &BTreeMap<OuterType, usize>)
                          -> BTreeMap<OuterType, usize> {
    let mut new_context: BTreeMap<OuterType, usize> = BTreeMap::new();
    for (key, value) in first_context.iter() {
        new_context.insert(key.to_owned(), *value);
    }

    for (key, value) in second_context.iter() {
        let _ = match key.to_owned() {
            OuterType::Str(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            OuterType::Date(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
            OuterType::Int(z) => {
                new_context.insert(OuterType::Str(z.to_string() + "_x"),
                                   *value + first_context.len())
            }
        };
    }
    new_context
}
