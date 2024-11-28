use crate::linearAlgebra::types::{Matrix, Vector};

pub fn dot(a: &Vector, b: &Vector) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn mat_vec_mul(matrix: &Matrix, vector: &Vector) -> Vector {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

pub fn add_vector(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x,y)| x + y).collect()
}


pub fn vector_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x,y)| x -y).collect()
}

pub fn scalar_mul(scalar: f64, vector: &Vector) -> Vector {
    vector.iter().map(|x| scalar * x).collect()
}


pub fn transpose(matrix: &Matrix) -> Matrix {
    let mut transposed = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

pub fn outer_product(a: &Vector, b: &Vector) -> Matrix {
    a.iter()
        .map(|&x| b.iter().map(|&y| x * y).collect())
        .collect()
}


pub fn scalar_mul_matrix(scalar: f64, matrix: &Matrix) -> Matrix {
    matrix
        .iter()
        .map(|row| row.iter().map(|&x| scalar * x).collect())
        .collect()
}

pub fn matrix_add(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| add_vector(row_a, row_b))
        .collect()
}