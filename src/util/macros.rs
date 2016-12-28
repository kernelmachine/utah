#[macro_export]
macro_rules! dataframe {
 {
    { $ ($column:expr => $data:expr),+}
} => { {
    let mut n : Vec<String> = Vec::new();
    $(
        n.push($column.to_owned());
    )*
    let a  = stack(Axis(1), &[ $(ArrayView2::from(&$data) ),+ ]).unwrap();
    let new_index : Vec<String> = (0..a.dim().0).map(|x| x.to_string()).collect();
    DataFrame::new(a).index(&new_index[..]).unwrap().columns(&n[..]).unwrap()

}
}
}

macro_rules! col {
    {
        $($element : expr),+
    } => {
        arr2(&[$($element)+]).t().to_owned()
    }
}
