use crate::data::{LLMDataItem, LLMDataSetBatch, LLMDataset, LLMDatasetBatcher};
use crate::model::{GPTModel, GPTModelConfig};
use crate::tokenizer::CharTokenizer;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::{PartialDataset, ShuffledDataset};
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::{prelude::*, tensor::backend::AutodiffBackend};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub steps: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub block_size: usize,
    #[config(default = 0.003)]
    pub learning_rate: f64,
    pub optimizer: AdamWConfig,
    pub model: GPTModelConfig,

    #[config(default = 42)]
    pub seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

impl<B: AutodiffBackend> TrainStep<LLMDataSetBatch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: LLMDataSetBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.data);
        let loss = self.loss(logits.clone(), batch.target.clone());

        let [b, t, c] = logits.dims();

        let logits = logits.reshape([b * t, c]);
        let targets = batch.target.reshape([b * t]);

        let output = ClassificationOutput::new(loss, logits, targets);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<LLMDataSetBatch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: LLMDataSetBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.data);
        let loss = self.loss(logits.clone(), batch.target.clone());

        let [b, t, c] = logits.dims();

        let logits = logits.reshape([b * t, c]);
        let targets = batch.target.reshape([b * t]);

        let output = ClassificationOutput::new(loss, logits, targets);

        output
    }
}

// Train a GPT model, using the config provided
// input_file: path to the file used for training
// artifact_dir: path to the directory where model checkpoints will be saved
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: &TrainingConfig,
    input_file: &str,
    device: B::Device,
) -> GPTModel<B> {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    let tokenizer = CharTokenizer::new();
    let text = std::fs::read_to_string(input_file).unwrap();

    let split = text.len() * 8 / 10;

    type PartialData = PartialDataset<ShuffledDataset<LLMDataset, LLMDataItem>, LLMDataItem>;

    let dts = ShuffledDataset::with_seed(LLMDataset::new(config.block_size, &text[..split], &tokenizer), config.seed);
    let dvs = ShuffledDataset::with_seed(LLMDataset::new(config.block_size, &text[split..], &tokenizer), config.seed);

    let data_train = PartialData::new(dts, 0, config.steps * config.batch_size);
    let data_valid = PartialData::new(dvs, 0, config.steps * config.batch_size);

    let batcher_train = LLMDatasetBatcher::<B>::new(device.clone());
    let batcher_valid = LLMDatasetBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)        
        .build(data_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(data_valid);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(1)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model should be saved successfully");

    model_trained
}

