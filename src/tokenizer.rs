use std::collections::HashMap;


pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}


pub struct CharTokenizer {
    vocab: HashMap<usize, char>,
    reversed_vocab: HashMap<char, usize>,    
}


impl CharTokenizer {
    pub fn new() -> Self {
        const CHARS : &str =  "\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789äüöÄÖÜß";

        let mut vocab = HashMap::new();
        let mut reversed_vocab = HashMap::new();
        CHARS.chars().enumerate().for_each(|(i, c)| {
            vocab.insert(i, c);
            reversed_vocab.insert(c, i);
        });

        Self { vocab, reversed_vocab }        
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.reversed_vocab[&c]).collect()
    }
    
    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&i| self.vocab[&i]).collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer_encode_decode() {
        let tokenizer = CharTokenizer::new();
        let s = "hii there";
        let v: Vec<usize> = tokenizer.encode(s);
        assert_eq!(s, tokenizer.decode(&v));

    }


    #[test]
    fn test_char_tokenizer_vocab_size() {
        let tokenizer = CharTokenizer::new();
        assert_eq!(tokenizer.vocab_size(),103);
    }

    #[test]
    fn test_tokenize_shakespeare() {
        let tokenizer = CharTokenizer::new();

        let input = std::fs::read_to_string("./gpt2_data/shakespeare.txt").unwrap();
        let tokens = tokenizer.encode(&input);

        println!("{:?}", &tokens[0..100]);
        assert_eq!(tokens.len(), input.len());
    }


}