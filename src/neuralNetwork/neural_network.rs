use rand::{random, Rng};
use crate::linearAlgebra::activation_functions::sigmoid_vector;
use crate::linearAlgebra::types::Vector;
use crate::linearAlgebra::types::Matrix;
use crate::linearAlgebra::vector::{add_vector, mat_vec_mul, matrix_add, outer_product, scalar_mul, scalar_mul_matrix, transpose, vector_sub};

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Matrix,
    weights_hidden_output: Matrix,
    bias_hidden: Vector,
    bias_output: Vector,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights_input_hidden = (0..hidden_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let weights_hidden_output = (0..output_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let bias_hidden = vec![0.0; hidden_size];
        let bias_output = vec![0.0; output_size];

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            learning_rate,
        }
    }

    pub fn forward(&self, input: &Vector) -> (Vector, Vector) {
        // Warstwa ukryta
        let hidden_input = add_vector(
            &mat_vec_mul(&self.weights_input_hidden, input),
            &self.bias_hidden,
        );
        let hidden_output = sigmoid_vector(&hidden_input);

        // Warstwa wyjściowa
        let final_input = add_vector(
            &mat_vec_mul(&self.weights_hidden_output, &hidden_output),
            &self.bias_output,
        );
        let final_output = sigmoid_vector(&final_input);

        (hidden_output, final_output)
    }

    // Propagacja wsteczna
    pub fn backward(
        &mut self,
        input: &Vector,
        hidden_output: &Vector,
        final_output: &Vector,
        target: &Vector,
    ) {
        // Błąd na wyjściu
        let output_errors = vector_sub(target, final_output);
        let output_gradients = output_errors
            .iter()
            .zip(final_output.iter())
            .map(|(error, output)| error * output * (1.0 - output))
            .collect::<Vector>();

        // Błąd w warstwie ukrytej
        let weights_hidden_output_t = transpose(&self.weights_hidden_output);
        let hidden_errors = mat_vec_mul(&weights_hidden_output_t, &output_gradients);
        let hidden_gradients = hidden_errors
            .iter()
            .zip(hidden_output.iter())
            .map(|(error, output)| error * output * (1.0 - output))
            .collect::<Vector>();

        // Aktualizacja wag i biasów
        let delta_weights_hidden_output = outer_product(&output_gradients, hidden_output);
        self.weights_hidden_output = matrix_add(
            &self.weights_hidden_output,
            &scalar_mul_matrix(self.learning_rate, &delta_weights_hidden_output),
        );
        self.bias_output = add_vector(
            &self.bias_output,
            &scalar_mul(self.learning_rate, &output_gradients),
        );

        let delta_weights_input_hidden = outer_product(&hidden_gradients, input);
        self.weights_input_hidden = matrix_add(
            &self.weights_input_hidden,
            &scalar_mul_matrix(self.learning_rate, &delta_weights_input_hidden),
        );
        self.bias_hidden = add_vector(
            &self.bias_hidden,
            &scalar_mul(self.learning_rate, &hidden_gradients),
        );
    }
}