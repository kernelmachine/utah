#[allow(unused_imports)]
use error_chain::*;
use csv;

error_chain! {
// The type defined for this error. These are the conventional
// and recommended names, but they can be arbitrarily chosen.
// It is also possible to leave this block out entirely, or
// leave it empty, and these names will be used automatically.
    types {
        Error, ErrorKind, ResultExt, Result;
    }



// Define additional `ErrorKind` variants. The syntax here is
// the same as `quick_error!`, but the `from()` and `cause()`
// syntax is not supported.
    errors {
        InvalidColumnName(t: String) {
            description("invalid column name")
            display("invalid column name: '{}'", t)
        }

        RowShapeMismatch {
            description("row shape mismatch.")
            display("row shape mismatch.")
        }
        IndexShapeMismatch(expected: String , actual: String) {
            description("index shape mismatch. Expected length: {}, Actual length: {}")
            display("index shape mismatch. Expected length: {}, Actual length: {}",  expected, actual)
        }
        ColumnShapeMismatch(expected: String, actual: String) {
            description("column shape mismatch. Expected length: {}, Actual length: {}")
            display("column shape mismatch. Expected length: {}, Actual length: {}",  expected, actual)
        }

        NoCommonValues {
            description("No common values.")
            display("Join failed. No common values.")
        }
        ParseError(t : String) {
            description("Parsing Error.")
            display("Read failed. Parsing Error. {}", t)
        }
    }


}
