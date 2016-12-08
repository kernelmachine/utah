
use error_chain::*;


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
        StackFail {
            description("stack operation failed.")
            display("stack operation failed.")
        }
        RowShapeMismatch {
            description("row shape mismatch.")
            display("row shape mismatch.")
        }
        ColumnShapeMismatch {
            description("column shape mismatch.")
            display("column shape mismatch.")
        }
        InvalidAxis {
            description("Invalid Axis.")
            display("Invalid Axis. Choose 0 or 1.")
        }
        NoCommonValues {
            description("No common values.")
            display("Join failed. No common values.")
        }
    }
}
