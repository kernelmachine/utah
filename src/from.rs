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

impl<'a, 'b> From<&'b &'a str> for OuterType {
    fn from(s: &'b &'a str) -> OuterType {
        OuterType::Str(s.to_string())
    }
}

impl<'a> From<&'a str> for OuterType {
    fn from(s: &'a str) -> OuterType {
        OuterType::Str(s.to_string())
    }
}

impl From<String> for OuterType {
    fn from(s: String) -> OuterType {
        OuterType::Str(s)
    }
}

impl<'a> From<&'a String> for OuterType {
    fn from(s: &'a String) -> OuterType {
        OuterType::Str(s.to_owned())
    }
}

impl<'a> From<&'a DateTime<UTC>> for OuterType {
    fn from(d: &'a DateTime<UTC>) -> OuterType {
        OuterType::Date(d.to_owned())
    }
}

impl From<i64> for OuterType {
    fn from(i: i64) -> OuterType {
        OuterType::Int(i)
    }
}
