use crate::model::{BigramModel, BigramModelConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::Backend, Int, Tensor};
use burn::{prelude::*, tensor::backend::AutodiffBackend};
use rand::prelude::*;

pub fn get_batch<B: Backend>(
    data: Tensor<B, 1, Int>,
    batch_size: usize,
    block_size: usize,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut rng = StdRng::seed_from_u64(42);

    let ix = (0..batch_size)
        .map(|_| rng.gen_range(0..data.dims()[0] - block_size))
        .collect::<Vec<_>>();

    let x = Tensor::stack::<2>(
        ix.iter()
            .map(|&i| data.clone().slice([i..i + block_size]))
            .collect::<Vec<_>>(),
        0,
    );

    let y = Tensor::stack::<2>(
        ix.iter()
            .map(|&i| data.clone().slice([i + 1..i + block_size + 1]))
            .collect::<Vec<_>>(),
        0,
    );

    (x, y)
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub block_size: usize,
    #[config(default = 0.003)]
    pub learning_rate: f64,
    pub optimizer: AdamWConfig,
    pub model: BigramModelConfig,
}

pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig,
    data_train: Tensor<B, 1, Int>,    
) -> BigramModel<B> {

     let device = data_train.device();
     let mut model: BigramModel<B> = config.model.init(&device);
     let mut optimizer = config.optimizer.init::<B, BigramModel<B>>();

     for _ in 0..config.epochs {
        // sample a batch
        let (x, y) = get_batch(data_train.clone(), config.batch_size, config.block_size);

        // forward pass
        let logits = model.forward(x.clone());
        let loss = model.loss(logits.clone(), y.clone());

        println!("loss: {:?}", loss.clone().into_scalar().elem::<f32>());

        // backward pass
        let grads = loss.backward();
        let grads= GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

     }

     model
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tokenizer::{CharTokenizer, Tokenizer};
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Int, Tensor},
    };

    #[test]
    fn train_test() {
        let tokenizer = CharTokenizer::new();
        let data = tokenizer.encode(
            std::fs::read_to_string("./gpt2_data/shakespeare.txt")
                .unwrap()
                .as_str(),
        );
        let n = data.len() * 9 / 10;
        //let total = data.len();

        let device = Default::default();
        let data = Tensor::<NdArray, 1, Int>::from_data(&data[..], &device);

        let train_data = data.clone().slice([0..n - 1]);
        //let test_data = data.clone().slice([n..total]);

        let block_size = 8;

        let x = train_data.clone().slice([0..block_size]);
        let y = train_data.clone().slice([1..block_size + 1]);
        (0..block_size).for_each(|i| {
            println!(
                "input:{} target:{}",
                x.clone().slice([0..i + 1]),
                y.clone().slice([i..i + 1])
            )
        });
    }

    #[test]
    fn get_batch_test() {
        let tokenizer = CharTokenizer::new();
        let data = tokenizer.encode(
            std::fs::read_to_string("./gpt2_data/shakespeare.txt")
                .unwrap()
                .as_str(),
        );

        let device = Default::default();
        let data = Tensor::<NdArray, 1, Int>::from_data(&data[..], &device);

        let (x, y) = get_batch(data, 10, 8);

        assert_eq!(x.shape().dims, [10, 8]);
        assert_eq!(y.shape().dims, [10, 8]);
    }

    #[test]
    fn train_test_2() {
        let tokenizer = CharTokenizer::new();

        let data = tokenizer.encode(
            std::fs::read_to_string("./gpt2_data/shakespeare.txt")
                .unwrap()
                .as_str(),
        );

        type MyBackend = Autodiff<NdArray>;

        let device = Default::default();
        let data = Tensor::<MyBackend, 1, Int>::from_data(&data[..], &device);
        let config = TrainingConfig {
            epochs: 100,
            batch_size: 32,
            block_size: 256,
            learning_rate: 3e-4,      
            optimizer: AdamWConfig::new(),      
            model: BigramModelConfig { vocab_size: tokenizer.vocab_size(), n_embd: 32, block_size: 256 },
        };

     
        let bm = train(&config,data);        
        let generated = bm.generate(vec![0usize], 100);

        println!("generated: {:?}", tokenizer.decode(&generated));

    }


}


