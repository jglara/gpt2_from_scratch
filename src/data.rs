use crate::tokenizer::Tokenizer;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Int, Tensor},
};

pub type LLMDataItem = Vec<usize>; // vec of token ids

pub struct LLMDataset {
    data: Vec<usize>,
    block_size: usize,
}

impl LLMDataset {
    pub fn new<T: Tokenizer>(block_size: usize, contents: &str, t: &T) -> Self {
        let data = t.encode(contents);

        Self { data, block_size }
    }
}

impl Dataset<LLMDataItem> for LLMDataset {
    fn get(&self, index: usize) -> Option<LLMDataItem> {
        Some(self.data[index..index + self.block_size+1].to_vec())
    }

    fn len(&self) -> usize {
        self.data.len() - (self.block_size+1)
    }
}

#[derive(Clone)]
pub struct LLMDatasetBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> LLMDatasetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct LLMDataSetBatch<B: Backend> {
    pub data: Tensor<B, 2, Int>, // (batch_size, block_size)
    pub target: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<LLMDataItem, LLMDataSetBatch<B>> for LLMDatasetBatcher<B> {
    fn batch(&self, data: Vec<LLMDataItem>) -> LLMDataSetBatch<B> {
        let x: Tensor<B, 2, Int> = Tensor::cat(
            data.iter()
                .map(|block| Tensor::<B, 1, Int>::from_data(&block[..block.len()-1], &self.device))
                .map(|t| t.unsqueeze())
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(&self.device);

        let y : Tensor<B, 2, Int> = Tensor::cat(
            data.iter()
                .map(|block| Tensor::<B, 1, Int>::from_data(&block[1..block.len()], &self.device))
                .map(|t| t.unsqueeze())
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(&self.device);

        LLMDataSetBatch { data: x, target: y }
    }
}
