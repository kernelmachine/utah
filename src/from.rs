use chrono::*;
use types::*;

impl From<f64> for InnerType {
    fn from(f: f64) -> InnerType {
        InnerType::Float(f)
    }
}

impl From<i64> for InnerType {
    fn from(i: i64) -> InnerType {
        InnerType::Int(i)
    }
}

impl<'a, 'b> From<&'b &'a str> for ColumnType {
    fn from(s: &'b &'a str) -> ColumnType {
        ColumnType::Str(s.to_string())
    }
}

impl<'a> From<&'a str> for ColumnType {
    fn from(s: &'a str) -> ColumnType {
        ColumnType::Str(s.to_string())
    }
}

impl From<String> for ColumnType {
    fn from(s: String) -> ColumnType {
        ColumnType::Str(s)
    }
}

impl<'a> From<&'a String> for ColumnType {
    fn from(s: &'a String) -> ColumnType {
        ColumnType::Str(s.to_owned())
    }
}

impl<'a> From<&'a DateTime<UTC>> for ColumnType {
    fn from(d: &'a DateTime<UTC>) -> ColumnType {
        ColumnType::Date(d.to_owned())
    }
}

impl From<i64> for ColumnType {
    fn from(i: i64) -> ColumnType {
        ColumnType::Int(i)
    }
}

impl<'a, 'b> From<&'b &'a str> for IndexType {
    fn from(s: &'b &'a str) -> IndexType {
        IndexType::Str(s.to_string())
    }
}

impl<'a> From<&'a str> for IndexType {
    fn from(s: &'a str) -> IndexType {
        IndexType::Str(s.to_string())
    }
}

impl<'a> From<&'a String> for IndexType {
    fn from(s: &'a String) -> IndexType {
        IndexType::Str(s.to_owned())
    }
}

impl From<String> for IndexType {
    fn from(s: String) -> IndexType {
        IndexType::Str(s)
    }
}

impl From<DateTime<UTC>> for IndexType {
    fn from(d: DateTime<UTC>) -> IndexType {
        IndexType::Date(d)
    }
}

impl From<i64> for IndexType {
    fn from(i: i64) -> IndexType {
        IndexType::Int(i)
    }
}
