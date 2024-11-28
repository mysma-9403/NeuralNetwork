use crate::linearAlgebra::types::Vector;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_vector(v: &Vector) -> Vector {
    v.iter().map(|&x| sigmoid(x)).collect()
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn sigmoid_derivative_vector(v: &Vector) -> Vector {
    v.iter().map(|&x| sigmoid_derivative(x)).collect()
}