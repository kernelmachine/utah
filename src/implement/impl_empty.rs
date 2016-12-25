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
