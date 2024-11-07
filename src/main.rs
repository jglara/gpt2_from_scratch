use std::path::PathBuf;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use burn::record::CompactRecorder;
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

    #[arg(long)]
    context: Option<String>,
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

fn main_with_backend<B: Backend>(
    cli: &Cli,
    tokenizer: &CharTokenizer,
    train_config: &TrainingConfig,
    device: B::Device,
) -> Result<()> {
    let bm: GPTModel<_> = cli.train.as_ref().map_or_else(
        || {
            train_config.model.init::<B>(&device).load_file(
                &format!("{}/model", cli.model),
                &CompactRecorder::new(),
                &device,
            )
        },
        |path| {
            train_config
                .model
                .init::<B>(&device)
                .load_file(&path, &CompactRecorder::new(), &device)
        },
    )?;

    if let Some(generate) = cli.generate {
        let ctxt = cli
            .context
            .as_ref()
            .map_or_else(|| vec![0usize], |c| tokenizer.encode(c.as_str()));
        let _generated = bm.generate(ctxt, 32, generate, tokenizer);
    }

    Ok(())
}

fn main() -> Result<()> {
    let tokenizer = CharTokenizer::new();
    let cli = Cli::parse();

    let conf = Conf::with_layers(&[Layer::Yaml(PathBuf::from(&cli.config))])?;

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
        main_with_backend::<MyBackend>(&cli, &tokenizer, &train_config, device)?;
    } else {
        type MyBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();
        main_with_backend::<MyBackend>(&cli, &tokenizer, &train_config, device)?;
    }

    Ok(())
}
