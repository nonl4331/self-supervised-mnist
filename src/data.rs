use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

pub const IMAGE_X: usize = 28;
pub const IMAGE_Y: usize = 28;
pub const INPUT_SIZE: usize = IMAGE_X * IMAGE_Y;

#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<3>,
    pub targets: Tensor<1, Int>,
}

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let images = items
            .iter()
            .map(|item| Tensor::<2>::from_data(TensorData::from(item.image), device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .collect();

        let targets = items.iter().map(|item| Tensor::<1, Int>::from_data([item.label as i64], device)).collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
