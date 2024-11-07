use std::path::PathBuf;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;

use burn::tensor::backend::AutodiffBackend;
use gpt2_from_scratch::learner::train;
use gpt2_from_scratch::learner::TrainingConfig;
use gpt2_from_scratch::model::{GPTModel, GPTModelConfig};
use gpt2_from_scratch::tokenizer::{CharTokenizer, Tokenizer};

use anyhow::{Context, Result};
use clap::Parser;
use twelf::{config, Layer};

#[derive(Parser, Default)]
struct Cli {
    #[clap(short, long, help = "Use GPU for training")]
    gpu: bool,

    /// Train the model
    #[clap(
        short,
        long,
        help = "Train the model. If this is not set, the model will be loaded from the model directory"
    )]
    train: Option<String>,

    /// Model directory
    #[clap(
        short,
        long,
        default_value = "./model/",
        help = "Directory to save the model and the artifacts"
    )]
    model: String,

    /// Config file
    #[clap(
        long,
        default_value = "./config.yaml",
        help = "Path to the config file with hyperparameters of the model architecture"
    )]
    config: String,

    /// Generate text
    #[clap(
        long,
        help = "Generate a number of tokens"
    )]
    generate: Option<usize>,

    /// Context for the generation
    #[clap(
        long,
        help = "Context for the generation"
    )]
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

fn train_and_generate<B: AutodiffBackend>(
    cli: &Cli,
    tokenizer: &CharTokenizer,
    train_config: &TrainingConfig,
    device: B::Device,
) -> Result<()> {
    let bm: GPTModel<_> = cli.train.as_ref().map_or_else(
        || {
            train_config
                .model
                .init::<B>(&device)
                .load_file(
                    &format!("{}/model", cli.model),
                    &CompactRecorder::new(),
                    &device,
                )
                .with_context(|| format!("failed to load model from {}/model", cli.model))
        },
        |path| Ok(train::<B>(&cli.model, &train_config, path, device.clone())),
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
        train_and_generate::<MyBackend>(&cli, &tokenizer, &train_config, device)?;
    } else {
        type MyBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();
        train_and_generate::<MyBackend>(&cli, &tokenizer, &train_config, device)?;
    }

    Ok(())
}
