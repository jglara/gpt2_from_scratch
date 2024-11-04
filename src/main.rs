use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use gpt2_from_scratch::learner::train;
use gpt2_from_scratch::learner::TrainingConfig;
use gpt2_from_scratch::model::{GPTModel, GPTModelConfig};
use gpt2_from_scratch::tokenizer::{CharTokenizer, Tokenizer};

use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    gpu: bool,

    #[arg(short, long)]
    train: bool,

    #[arg(short, long)]
    out: String,

    
}

fn main() {
    let tokenizer = CharTokenizer::new();

    let config = TrainingConfig {
        steps: 100,
        batch_size: 64,
        block_size: 256,
        learning_rate: 3e-4,
        optimizer: AdamWConfig::new(),
        model: GPTModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: 384,
            n_heads: 6,
            n_layers: 6,
            block_size: 256,
        },
        seed: 42,
    };

    let cli = Cli::parse();

    if cli.gpu {
        type MyBackend = Autodiff<Wgpu>;
        let device = WgpuDevice::default();

        let bm: GPTModel<_> = if cli.train {
            train::<MyBackend>(&format!("{}", cli.out), &config, "./gpt2_data/shakespeare.txt", device)
        } else {
            config
                .model
                .init(&device)
                .load_file(&format!("{}/model", cli.out), &CompactRecorder::new(), &device)
                .expect("File not found")
        };

        let generated = bm.generate(vec![0usize], 100);

        tokenizer
            .decode(&generated)
            .lines()
            .for_each(|l| println!("{}", l));

    } else {
        type MyBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let bm: GPTModel<_> = if cli.train {
            train::<MyBackend>(&format!("{}", cli.out), &config, "./gpt2_data/shakespeare.txt", device)
        } else {
            config
                .model
                .init(&device)
                .load_file(&format!("{}/model", cli.out), &CompactRecorder::new(), &device)
                .expect("File not found")
        };
        
        let generated = bm.generate(vec![0usize], 100);

        tokenizer
            .decode(&generated)
            .lines()
            .for_each(|l| println!("{}", l));
        
    }
}
