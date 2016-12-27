use util::traits::Empty;
use std::f64::NAN;

impl Empty<f64> for f64 {
    fn empty() -> f64 {
        NAN
    }
    fn is_empty(&self) -> bool {
        self.is_nan()
    }
}

impl Empty<i32> for i32 {
    fn empty() -> i32 {
        0
    }
    fn is_empty(&self) -> bool {
        *self == 0
    }
}

impl Empty<Option<i32>> for Option<i32> {
    fn empty() -> Option<i32> {
        None
    }
    fn is_empty(&self) -> bool {
        self.is_none()
    }
}


impl Empty<Option<f64>> for Option<f64> {
    fn empty() -> Option<f64> {
        None
    }
    fn is_empty(&self) -> bool {
        self.is_none()
    }
}
