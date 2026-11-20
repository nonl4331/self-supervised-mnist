use burn::{
    data::{
        dataloader::DataLoaderBuilder, dataset::{transform::PartialDataset, vision::{MnistDataset, MnistItem}},
    }, optim::AdamConfig, prelude::*, train::{ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep, metric::{AccuracyMetric, LossMetric}},
};
use std::{fs, sync::Arc};

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::Model,
};

const ARTIFACT_DIR: &str = "/tmp/ssm";

#[derive(Config, Debug)]
pub struct TrainingConf {
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(defualt = 256)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 0)]
    pub seed: u64,
    pub optim: AdamConfig,
}

pub fn run(device: Device) {
    let conf = TrainingConf::new(AdamConfig::new());

    device.seed(conf.seed);
    let autodiff_device = device.clone().autodiff();

    let model = Model::new(&autodiff_device);

    let dataset = Arc::new(MnistDataset::train());
    let dataset_train: PartialDataset<Arc<_>, MnistItem> = PartialDataset::new(dataset.clone(), 0, 55000);
    let dataset_val: PartialDataset<Arc<_>, MnistItem> = PartialDataset::new(dataset.clone(), 55000, 60000);
    let dataset_test = MnistDataset::test();

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(conf.batch_size)
        .shuffle(conf.seed)
        .num_workers(conf.num_workers)
        .build(dataset_train);

    let dataloader_val = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(conf.batch_size)
        .shuffle(conf.seed)
        .num_workers(conf.num_workers)
        .build(dataset_val);

    let dataloader_test = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(conf.batch_size)
        .shuffle(conf.seed)
        .num_workers(conf.num_workers)
        .build(dataset_test);

    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_val).metrics((AccuracyMetric::new(), LossMetric::new())).with_default_checkpointers().num_epochs(conf.num_epochs).summary();

    let result = training.launch(Learner::new(model, conf.optim.init(), 1e-3));
}

impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.foward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.foward_classification(batch.images, batch.targets)
    }
}
