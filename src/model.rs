use {
    burn::{
        nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig},
        prelude::*,
        tensor::activation,
    },
    nn::{Linear, LinearConfig},
    rand::{distributions::WeightedIndex, prelude::*},
};

#[derive(Config, Debug)]
pub struct BigramModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub block_size: usize,
}

#[derive(Debug, Module)]
pub struct BigramModel<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    lm_head: Linear<B>,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        BigramModel {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.n_embd).init(device),
            lm_head: LinearConfig::new(self.n_embd, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, t] = input.dims();
        let tok_emb = self.token_embedding.forward(input.clone());
        let pos_emb = self
            .position_embedding
            .forward(Tensor::arange(0..(t as i64), &input.device()).unsqueeze());

        let x = tok_emb + pos_emb;
        let logits = self.lm_head.forward(x);

        logits
    }

    pub fn loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims();

        let logits = logits.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets);

        loss
    }

    pub fn generate(&self, context: Vec<usize>, max_tokens: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(42);
        let device = B::Device::default();

        let mut toks = context.clone();

        for _ in 0..max_tokens {
            let logits = self.forward(
                Tensor::<B, 1, Int>::from_data(TensorData::from(&toks[..]), &device).unsqueeze(),
            );
            // focus only on the last timestep
            let [b, t, c] = logits.dims();

            let slice = logits.slice([0..b, t - 1..t, 0..c]).flatten::<1>(0, 2);
            let probs: Vec<f32> = activation::softmax(slice, 0).into_data().to_vec().unwrap();

            let distribution = WeightedIndex::new(&probs[..probs.len() - 1]).unwrap();

            let prediction = distribution.sample(&mut rng) as usize;
            toks.push(prediction);
        }

        toks
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use super::*;
    use crate::tokenizer::{CharTokenizer, Tokenizer};
    use crate::train::get_batch;
    use burn::backend::Autodiff;
    use burn::{
        backend::NdArray,
        tensor::{Int, Tensor},
    };

    #[test]
    fn bigram_model_test() {
        let tokenizer = CharTokenizer::new();
        let data = tokenizer.encode(
            std::fs::read_to_string("./gpt2_data/shakespeare.txt")
                .unwrap()
                .as_str(),
        );

        let device = Default::default();
        let data = Tensor::<NdArray, 1, Int>::from_data(&data[..], &device);

        let (x, y) = get_batch(data, 10, 8);

        let bm = BigramModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 32,
            block_size: 64,
            
        }
        .init::<NdArray>(&device);
        let logits = bm.forward(x);
        let loss = bm.loss(logits.clone(), y);

        println!("logits: {:?},loss: {:?}", logits.dims(), loss);
    }

    #[test]
    fn generate_test() {
        let tokenizer = CharTokenizer::new();
        let toks = vec![0usize];

        let bm = BigramModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 32,
            block_size: 64,
        }
        .init::<NdArray>(&Default::default());
        let generated = bm.generate(toks, 100);

        println!("generated: {:?}", tokenizer.decode(&generated));
    }

    #[test]
    fn self_attention_test() {
        type MyBackend = Autodiff<NdArray>;
        let device = Default::default();

        let (b, t, c) = (4, 8, 2);
        let x = Tensor::<MyBackend, 2, Float>::random(
            Shape::new([t, c]),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let mask = Tensor::<MyBackend, 2, Bool>::tril_mask([t, t], 0, &device);
        print!("mask: {:?}", mask);
        let wei = Tensor::<MyBackend, 2>::zeros([t, t], &device);
        print!("wei: {:?}", wei);
        let wei = wei.mask_fill(mask, f32::NEG_INFINITY);
        print!("wei: {:?}", wei);
        let wei = activation::softmax(wei, 1);
        print!("wei: {:?}", wei);

        let x = wei.matmul(x);
        println!("x: {:?}", x);
    }
}
