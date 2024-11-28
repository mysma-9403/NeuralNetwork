use std::collections::HashMap;

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .collect()
}

pub fn build_vocabulary(texts: &[&str]) -> HashMap<String, usize> {
    let mut vocab = HashMap::new();
    let mut index = 0;
    for &text in texts {
        for token in tokenize(text) {
            if !vocab.contains_key(&token) {
                vocab.insert(token.to_string(), index);
                index += 1;
            }
        }
    }
    vocab
}