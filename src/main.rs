use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::optim::AdamWConfig;
use gpt2_from_scratch::model::{BigramModel, BigramModelConfig};
use gpt2_from_scratch::tokenizer::{CharTokenizer, Tokenizer};
use gpt2_from_scratch::learner::train;
use gpt2_from_scratch::learner::TrainingConfig;



fn main() {
    
    type MyBackend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();
    let tokenizer = CharTokenizer::new();

    let config = TrainingConfig {
        steps: 100,
        batch_size: 32,
        block_size: 256,
        learning_rate: 3e-4,
        optimizer: AdamWConfig::new(),
        model: BigramModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 32,
            n_heads: 4,
            n_layers: 4,
            block_size: 256,
        },
        seed: 42,
    };

    let bm: BigramModel<_> =
        train::<MyBackend>("./models", &config, "./gpt2_data/shakespeare.txt", device);

    let generated = bm.generate(vec![0usize], 100);

    println!("generated: {:?}", tokenizer.decode(&generated));
}
