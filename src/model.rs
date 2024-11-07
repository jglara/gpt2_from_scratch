use {
    crate::tokenizer::Tokenizer, burn::{
        nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig},
        prelude::*,
        tensor::activation,
    }, nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu}, rand::{distributions::WeightedIndex, prelude::*}, std::io::{self, Write}
};

#[derive(Config, Debug)]
pub struct GPTModelConfig {
    pub vocab_size: usize, // number of tokens in the vocabulary
    pub n_embd: usize,     // embedding dimension
    pub block_size: usize, // number of tokens to use in each context
    pub n_layers: usize,   // number of blocks / layers in the model
    pub n_heads: usize,    // number of heads in each self attention block
}

#[derive(Debug, Module)]
pub struct GPTModel<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl GPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPTModel<B> {
        GPTModel {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.n_embd).init(device),
            blocks: (0..self.n_layers)
                .map(|_| BlockConfig::new(self.n_embd, self.n_heads).init(device))
                .collect(),
            norm: LayerNormConfig::new(self.n_embd).init(device),
            lm_head: LinearConfig::new(self.n_embd, self.vocab_size).init(device),
        }
    }
}

// GPT Model
// input: [b,t] b: batch size t: block_size (context size)
// output: [b,t,vocab_size]
// 
impl<B: Backend> GPTModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, t] = input.dims();
        let tok_emb = self.token_embedding.forward(input.clone()); // [b,t,n_embd]
        let pos_emb = self
            .position_embedding
            .forward(Tensor::arange(0..(t as i64), &input.device()).unsqueeze());

        let x = tok_emb + pos_emb; // [b,t,n_embd]
        let x = self.blocks.iter().fold(x, |x, b| b.forward(x)); // [b,t,n_embd]
        let x = self.norm.forward(x); // [b,t,n_embd]
        let logits = self.lm_head.forward(x); // [b,t,vocab_size]

        logits
    }

    // loss function for training (CrossEntropyLoss)
    //
    // input: logits [b,t,vocab_size] target: [b,t]
    // output: [1]
    pub fn loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims();

        let logits = logits.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets);

        loss
    }

    // Generation of tokens using the given context and model
    // max_tokens: number of tokens to generate
    pub fn generate(
        &self,
        context: Vec<usize>,
        context_length: usize,
        max_tokens: usize,
        tokenizer: &impl Tokenizer,
    ) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(42);
        let device = B::Device::default();

        let mut toks = context.clone();

        println!("{}", tokenizer.decode(&context));
        for _ in 0..max_tokens {
            
       
            let input =
                &toks[std::cmp::max(0, toks.len() as isize - context_length as isize) as usize..];
            
            
            let logits = self.forward(
                Tensor::<B, 1, Int>::from_data(TensorData::from(input), &device).unsqueeze(),
            );
            
            // focus only on the last timestep
            let [b, t, c] = logits.dims();

            
            let slice = logits.slice([0..b, t - 1..t, 0..c]).flatten::<1>(0, 2);
            let probs: Vec<f32> = activation::softmax(slice, 0).into_data().to_vec().unwrap();
            
            let distribution = WeightedIndex::new(&probs[..probs.len() - 1]).unwrap();
            
            let prediction = distribution.sample(&mut rng) as usize;
           
            print!("{}", tokenizer.decode(&[prediction]));
            io::stdout().flush().unwrap();
            
            
            toks.push(prediction);
        
            
        }
        println!("");

        toks
    }
}

///// Single Head Attention Model
/// 
#[derive(Config, Debug)]
pub struct SingleHeadModelConfig {
    pub n_embd: usize,
    pub head_size: usize,
}

// Single Head Attention Model
// https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s
#[derive(Debug, Module)]
pub struct SingleHeadModel<B: Backend> {
    pub key: Linear<B>,
    pub query: Linear<B>,
    pub value: Linear<B>,
}

impl<B: Backend> SingleHeadModel<B> {
    // input: [b,t,c]. b: batch size t: block_size (context size) c: head_size
    // output: [b,t,c]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, t, c] = input.dims();
        let k = self.key.forward(input.clone()); // [b,t,head_size]
        let q = self.query.forward(input.clone()); // [b,t,head_size]

        let wei = q.matmul(k.transpose()); // [b,t,head_size] @ [b,head_size,t] = [b,t,t]
        let wei = wei.div_scalar((c as f32).sqrt()); // normalize to keep variance and make softmax later work better

        // triangular mask to avoid looking ahead. https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2831s
        // [[1, 0, 0],
        //  [1, 1, 0],
        //  [1, 1, 1]]

        let mask = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &input.device()).unsqueeze(); // [1,t,t]
        let wei = wei.mask_fill(mask, f32::NEG_INFINITY);

        let wei = activation::softmax(wei, 2);
        let v = self.value.forward(input.clone()); // [b,t,head_size]

        let out = wei.matmul(v); // [b,t,t] @ [b,t,head_size] = [b,t,head_size]
        out
    }
}

impl SingleHeadModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SingleHeadModel<B> {
        SingleHeadModel {
            key: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
            query: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(self.n_embd, self.head_size)
                .with_bias(false)
                .init(device),
        }
    }
}

////// Multi Head Attention Model
/// https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s

#[derive(Config, Debug)]
pub struct MultiHeadModelConfig {
    pub n_heads: usize,
    pub n_embd: usize,
    pub head_size: usize,
    #[config(default = 0.2)]
    pub dropout: f64,
}

#[derive(Debug, Module)]
pub struct MultiHeadModel<B: Backend> {
    pub heads: Vec<SingleHeadModel<B>>,
    pub proj: Linear<B>,
    pub dropout: Dropout,
}

impl MultiHeadModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadModel<B> {
        MultiHeadModel {
            heads: (0..self.n_heads)
                .map(|_| SingleHeadModelConfig::new(self.n_embd, self.head_size).init(device))
                .collect(),
            proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> MultiHeadModel<B> {
    // input: [b,t,c]. b: batch size t: block_size (context size) c: head_size
    // output: [b,t,c]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Run n_heads single head models in parallel
        let outs = self
            .heads
            .iter()
            .map(|h| h.forward(input.clone()))
            .collect::<Vec<_>>();

        let x = Tensor::cat(outs, 2);
        let x = self.proj.forward(x);
        x
    }
}

///// Positional Feeed Forward Model
///
/// https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s
#[derive(Config, Debug)]
pub struct PositionalFeedForwardModelConfig {
    pub n_embd: usize,
    #[config(default = 0.2)]
    pub dropout: f64,
}

#[derive(Debug, Module)]
pub struct PositionalFeedForwardModel<B: Backend> {
    pub layer_1: Linear<B>,
    pub act_1: Relu, // could also be a Gelu activation, like in GPT2
    pub layer_2: Linear<B>,
    pub dropout: Dropout,
}

impl PositionalFeedForwardModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalFeedForwardModel<B> {
        PositionalFeedForwardModel {
            layer_1: LinearConfig::new(self.n_embd, 4 * self.n_embd).init(device),
            act_1: Relu::new(),
            layer_2: LinearConfig::new(4 * self.n_embd, self.n_embd).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> PositionalFeedForwardModel<B> {
    // input: [b,t,c]. b: batch size t: block_size (context size) c: head_size
    // output: [b,t,c]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.layer_1.forward(input.clone());
        let x = self.act_1.forward(x);
        let x = self.layer_2.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}

/// Block Model
///
///

#[derive(Config, Debug)]
pub struct BlockConfig {
    n_embd: usize,
    n_heads: usize,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub norm_1: LayerNorm<B>,
    pub attn: MultiHeadModel<B>,
    pub norm_2: LayerNorm<B>,
    pub ffwd: PositionalFeedForwardModel<B>,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        assert!(self.n_embd % self.n_heads == 0); // n_embsd must be divisible by n_heads to MultiHeadModel
        let head_size = self.n_embd / self.n_heads;

        Block {
            norm_1: LayerNormConfig::new(self.n_embd).init(device),
            attn: MultiHeadModelConfig::new(self.n_heads, self.n_embd, head_size).init(device),
            norm_2: LayerNormConfig::new(self.n_embd).init(device),
            ffwd: PositionalFeedForwardModelConfig::new(self.n_embd).init(device),
        }
    }
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm_1.forward(input.clone());
        let x = x.clone() + self.attn.forward(x);
        let x = self.norm_2.forward(x);
        let x = x.clone() + self.ffwd.forward(x);
        x
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
        //type MyBackend = Autodiff<NdArray>;

        let tokenizer = CharTokenizer::new();
        let data = tokenizer.encode(
            std::fs::read_to_string("./gpt2_data/shakespeare.txt")
                .unwrap()
                .as_str(),
        );

        let device = Default::default();
        let data = Tensor::<NdArray, 1, Int>::from_data(&data[..], &device);

        let (x, y) = get_batch(data, 10, 8);

        let bm = GPTModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 32,
            n_heads: 4,
            n_layers: 4,
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

        let bm = GPTModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 32,
            n_heads: 4,
            n_layers: 4,
            block_size: 64,
        }
        .init::<NdArray>(&Default::default());
        let generated = bm.generate(toks, 10, 100, &tokenizer);

        println!("generated: {:?}", tokenizer.decode(&generated));
    }

    #[test]
    fn self_attention_test() {
        type MyBackend = Autodiff<NdArray>;
        let device = Default::default();

        let (_b, t, c) = (4, 8, 32);
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
