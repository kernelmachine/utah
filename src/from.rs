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

impl From<String> for ColumnType {
    fn from(s: String) -> ColumnType {
        ColumnType::Str(s)
    }
}

impl From<DateTime<UTC>> for ColumnType {
    fn from(d: DateTime<UTC>) -> ColumnType {
        ColumnType::Date(d)
    }
}

impl From<i64> for ColumnType {
    fn from(i: i64) -> ColumnType {
        ColumnType::Int(i)
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
