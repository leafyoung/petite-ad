pub fn f(x: f64) -> f64 {
    (x.sin().sin()).exp()
}

pub fn f_prime(x: f64) -> f64 {
    f(x) * x.sin().cos() * x.cos()
}
