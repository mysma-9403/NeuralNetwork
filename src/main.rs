mod linearAlgebra;
mod textProcessing;
mod neuralNetwork;

use textProcessing::tokenizer::build_vocabulary;
use textProcessing::vectorizer::text_to_vector;
use crate::neuralNetwork::neural_network::NeuralNetwork;
use crate::textProcessing::vectorizer::vector_to_text;

fn main() {


    let texts = vec![
        ("What is your name?", "My name is RustBot."),
        ("How are you?", "I am fine, thank you."),
        ("What is Rust?", "Rust is a programming language focused on safety and performance."),
        ("Who created Rust?", "Rust was originally created by Graydon Hoare."),
        ("What is your favorite programming language?", "I love Rust!"),
        ("How do you learn programming?", "By practicing and solving problems."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the largest planet?", "Jupiter is the largest planet in the Solar System."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
        ("What is the square root of 16?", "The square root of 16 is 4."),
        ("How do I declare a variable in Rust?", "Use the `let` keyword to declare a variable."),
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        ("What is AI?", "AI stands for Artificial Intelligence."),
        ("What is 5 times 5?", "5 times 5 equals 25."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain."),
        ("Who is the current president of the United States?", "It depends on the current year."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius."),
        ("What is the Python programming language used for?", "Python is used for web development, data analysis, and more."),
        ("What is the first programming language?", "Fortran is considered one of the first programming languages."),
        ("Who discovered gravity?", "Isaac Newton is credited with discovering gravity."),
        ("What is the smallest prime number?", "The smallest prime number is 2."),
        ("What is the currency of the United Kingdom?", "The currency of the UK is the pound sterling."),
        ("What is an algorithm?", "An algorithm is a step-by-step procedure for solving a problem."),
        ("What is recursion?", "Recursion is when a function calls itself."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
        ("What is the Eiffel Tower?", "The Eiffel Tower is a famous landmark in Paris."),
        ("What is your favorite food?", "I don't eat, but I hear pizza is popular."),
        ("What is binary?", "Binary is a base-2 numeral system used in computing."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("What is the smallest country?", "Vatican City is the smallest country."),
        ("What is the fastest animal?", "The peregrine falcon is the fastest animal."),
        ("What is a compiler?", "A compiler translates code into executable programs."),
        ("What is the color of the sky?", "The sky is blue."),
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("What is the longest river?", "The Nile is often considered the longest river."),
        ("What is the freezing point of water?", "The freezing point of water is 0 degrees Celsius."),
        ("What is a chatbot?", "A chatbot is a computer program that simulates conversation."),
        ("What is the moon?", "The moon is Earth's natural satellite."),
        ("What is the Fibonacci sequence?", "It is a sequence where each number is the sum of the two preceding ones."),
        ("What is the square of 7?", "The square of 7 is 49."),
        ("What is the meaning of life?", "42, according to The Hitchhiker's Guide to the Galaxy."),
        ("What is gravity?", "Gravity is a force that attracts objects toward each other."),
        ("What is the tallest building?", "The tallest building is the Burj Khalifa in Dubai."),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
        ("What is the internet?", "The internet is a global network of computers."),
        ("What is the sum of 10 and 20?", "The sum of 10 and 20 is 30."),
        ("What is the square of 8?", "The square of 8 is 64."),
        ("What is the capital of Spain?", "The capital of Spain is Madrid."),
        ("What is the Pythagorean theorem?", "It states that a² + b² = c² in a right triangle."),
        ("What is an atom?", "An atom is the basic unit of matter."),
        ("What is DNA?", "DNA is the molecule that carries genetic information."),
        ("What is a planet?", "A planet is a celestial body orbiting a star."),
        ("What is photosynthesis?", "Photosynthesis is how plants convert sunlight into energy."),
        ("What is your purpose?", "To assist and provide information."),
        ("What is a function in programming?", "A function is a reusable block of code."),
        ("What is the capital of Russia?", "The capital of Russia is Moscow."),
        ("What is the speed of sound?", "The speed of sound is about 343 meters per second."),
        ("What is the capital of Australia?", "The capital of Australia is Canberra."),
        ("What is a vector in Rust?", "A vector is a dynamic array in Rust."),
        ("What is HTML?", "HTML stands for HyperText Markup Language."),
        ("What is CSS?", "CSS stands for Cascading Style Sheets."),
        ("What is JavaScript?", "JavaScript is a programming language for the web."),
        ("What is a database?", "A database is a collection of organized data."),
        ("What is SQL?", "SQL stands for Structured Query Language."),
        ("What is Rust?", "Rust is a systems programming language."),
        ("What is a loop in programming?", "A loop is a construct for repeating code."),
        ("What is recursion?", "Recursion is when a function calls itself."),
        ("What is the largest desert?", "The Sahara Desert is the largest hot desert."),
        ("What is a neural network?", "A neural network is a system modeled after the human brain."),
        ("What is the capital of China?", "The capital of China is Beijing."),
        ("What is the largest continent?", "Asia is the largest continent."),
        ("What is the Milky Way?", "The Milky Way is the galaxy containing our Solar System."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("What is a byte?", "A byte is 8 bits of data."),
        ("What is the fastest car?", "The Bugatti Chiron is one of the fastest cars."),
        ("What is a star?", "A star is a luminous celestial body."),
        ("What is the smallest particle?", "Quarks are among the smallest known particles."),
        ("What is an API?", "An API is an Application Programming Interface."),
        ("What is a cloud?", "In computing, it's a system of servers."),
        ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
        ("What is a black hole?", "A black hole is a region of spacetime with strong gravity."),
        ("What is Python?", "Python is a versatile programming language."),
        ("What is the universe?", "The universe is everything that exists."),
        ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
        ("What is the Great Wall of China?", "It is a series of fortifications built in China."),
        ("What is the capital of South Africa?", "South Africa has three capitals: Pretoria, Cape Town, and Bloemfontein."),
        ("What is the capital of Saudi Arabia?", "The capital of Saudi Arabia is Riyadh."),
        ("What is E=mc²?", "It is Einstein's equation relating energy and mass."),
        ("What is Newton's first law?", "An object at rest stays at rest unless acted on by a force."),
        ("What is energy?", "Energy is the ability to do work."),
    ];

    let mut all_texts = Vec::new();
    for (question, answer) in texts.iter() {
        all_texts.push(*question);
        all_texts.push(*answer);
    }

    let vocab = build_vocabulary(&all_texts);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for (question, answer) in texts.iter() {
        inputs.push(text_to_vector(question, &vocab));
        targets.push(text_to_vector(answer, &vocab));
    }

    let input_size = vocab.len();
    let hidden_size = 20;
    let output_size = vocab.len();
    let learning_rate = 0.1;

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size, learning_rate);

    for epoch in 0..1000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let (hidden_output, final_output) = nn.forward(input);
            nn.backward(input, &hidden_output, &final_output, target);
        }
        if epoch % 100 == 0 {
            println!("Epoka {} zakończona", epoch);
        }
    }

    let test_question = "Write me something interesintg?";
    let test_input = text_to_vector(test_question, &vocab);
    println!("{:?}", test_input);
    let (_hidden_output, final_output) = nn.forward(&test_input);

    let response = vector_to_text(&final_output, &vocab);
    println!("Pytanie: {}", test_question);
    println!("Odpowiedź: {}", response);
}
