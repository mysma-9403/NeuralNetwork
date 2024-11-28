use std::collections::HashMap;
use crate::linearAlgebra::types::Vector;
use crate::textProcessing::tokenizer::tokenize;

pub fn text_to_vector(text: &str, vocab: &HashMap<String, usize>) -> Vector {
    let mut vector = vec![0.0; vocab.len()];
    for token in tokenize(text) {
        if let Some(&index) = vocab.get(&token) {
            vector[index] = 1.0;
        }
    }
    vector
}

pub fn vector_to_text(vector: &Vector, vocab: &HashMap<String, usize>) -> String {
    let mut words = Vec::new();
    for (word, &index) in vocab {
        if vector[index] > 0.5 {
            words.push(word.clone());
        }
    }
    words.join(" ")
}
