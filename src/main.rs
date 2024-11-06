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

use anyhow::Result;
use clap::Parser;
use twelf::{config, Layer};

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    gpu: bool,

    #[arg(short, long)]
    train: Option<String>,

    #[arg(short, long)]
    model: String,

    #[arg(long)]
    config: String,

    #[arg(long)]
    generate: Option<usize>,
}

#[config]
#[derive(Default)]
struct Conf {
    steps: usize,
    batch_size: usize,
    n_ctxt: usize,
    learning_rate: f64,
    n_embd: usize,
    n_heads: usize,
    n_layers: usize,
}

fn main() -> Result<()> {
    let tokenizer = CharTokenizer::new();
    let cli = Cli::parse();

    let conf = Conf::with_layers(&[Layer::Yaml(cli.config.into())])?;

    let train_config = TrainingConfig {
        steps: conf.steps,
        batch_size: conf.batch_size,
        block_size: conf.n_ctxt,
        learning_rate: conf.learning_rate,
        optimizer: AdamWConfig::new(),
        model: GPTModelConfig {
            vocab_size: tokenizer.vocab_size(),
            n_embd: conf.n_embd,
            n_heads: conf.n_heads,
            n_layers: conf.n_layers,
            block_size: conf.n_ctxt,
        },
        seed: 42,
    };

    if cli.gpu {
        type MyBackend = Autodiff<Wgpu>;
        let device = WgpuDevice::default();

        let bm: GPTModel<_> = if cli.train.is_some() {
            train::<MyBackend>(
                &format!("{}", cli.model),
                &train_config,
                &cli.train.unwrap(),
                device.clone(),
            )
        } else {
            train_config.model.init(&device).load_file(
                &format!("{}/model", cli.model),
                &CompactRecorder::new(),
                &device,
            )?
        };

        if let Some(generate) = cli.generate {
          
                let _generated = bm.generate(vec![0usize], 32, generate, &tokenizer);

               
           
           
        }
    } else {
        type MyBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let bm: GPTModel<_> = if cli.train.is_some() {
            train::<MyBackend>(
                &format!("{}", cli.model),
                &train_config,
                &cli.train.unwrap(),
                device,
            )
        } else {
            train_config.model.init(&device).load_file(
                &format!("{}/model", cli.model),
                &CompactRecorder::new(),
                &device,
            )?
        };

        if let Some(generate) = cli.generate {
            let _generated = bm.generate(vec![0usize], 32, generate, &tokenizer);

         
        }
    }

    Ok(())
}
