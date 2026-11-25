use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            transform::PartialDataset,
            vision::{MnistDataset, MnistItem},
        },
    }, module::ModuleDisplay, optim::{AdamConfig, GradientsParams}, prelude::*, train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, IterationSpeedMetric, LossMetric},
    },
};
use std::sync::Arc;

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::*,
};

const ARTIFACT_DIR: &str = "/tmp/ssm";

#[derive(Config, Debug)]
pub struct TrainingConf {
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(defualt = 1024)]
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

    //let model = Model::new(&autodiff_device);
    let model2a_backbone = Model2Backbone::new(&autodiff_device);
    let model2b_backbone = Model2Backbone::new(&autodiff_device);
    let mut model2_backbone = JointEmbeddingModel::new(model2a_backbone, model2b_backbone);
    //let mut model2 = Model2::new(&autodiff_device);

    let dataset = Arc::new(MnistDataset::train());
    let dataset_train: PartialDataset<Arc<_>, MnistItem> =
        PartialDataset::new(dataset.clone(), 0, 55000);
    let dataset_val: PartialDataset<Arc<_>, MnistItem> =
        PartialDataset::new(dataset.clone(), 55000, 60000);
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

    // FFN training
    /*
    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_val).metrics((AccuracyMetric::new(), LossMetric::new(), IterationSpeedMetric::new())).with_default_checkpointers().num_epochs(conf.num_epochs).summary();
    let result = training.launch(Learner::new(model, conf.optim.init(), 1e-3));
    */

    println!("doing pretraining");
    // joint embedding pretraining of backbone
    let mut optim = conf.optim.init();
    for epoch in 1..conf.num_epochs + 1 {
        for (iter, batch) in dataloader_train.iter().map(Result::unwrap).enumerate() {
            let corr = corrupt_batch(batch.images.clone());

            let (za, zb) = model2_backbone.out(batch.images, corr);

            let (av, am) = za.clone().var_mean(0);
            let za_norm = (za - am) / av.sqrt();

            let (bv, bm) = zb.clone().var_mean(0);
            let zb_norm = (zb - bm) / bv.sqrt();

            let cross_corr = za_norm.t().matmul(zb_norm) / conf.batch_size as f32;

            let c_diff = (cross_corr - Tensor::eye(10, &autodiff_device)).powi_scalar(2);

            // assume lambda = 1
            let loss = c_diff.sum();
            let grads = loss.backward();
            println!("epoch {epoch} | iter {iter} | loss: {:.5}", loss.into_scalar::<f32>());

            let grads = GradientsParams::from_grads(grads, &model2_backbone);
            model2_backbone = optim.step(1e-3, model2_backbone, grads);
        }
        println!("epoch {epoch} done");
    }
    println!("finished pretraining");

    let mut model2 = JointEmbeddingPredictor::new(&autodiff_device, model2_backbone);
    model2.backbone = model2.backbone.freeze();

    // linear probe
    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_val)
        .metrics((
            AccuracyMetric::new(),
            LossMetric::new(),
            IterationSpeedMetric::new(),
        ))
        .with_default_checkpointers()
        .num_epochs(conf.num_epochs)
        .summary();
    let _result = training.launch(Learner::new(model2, conf.optim.init(), 1e-3));
}

fn corrupt_batch(batch: Tensor<3>) -> Tensor<3> {
    let device = batch.device();
    let shape = batch.shape();
    let mask = Tensor::<3>::random(shape, burn::tensor::Distribution::Bernoulli(0.5), &device);
    batch * mask
}

impl TrainStep for MnistMLPBaseline {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.foward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for MnistMLPBaseline {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.foward_classification(batch.images, batch.targets)
    }
}

impl<M: Module + ModelForward + ModuleDisplay> TrainStep for JointEmbeddingPredictor<M> {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.foward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<M: Module + ModelForward + ModuleDisplay> InferenceStep for JointEmbeddingPredictor<M> {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.foward_classification(batch.images, batch.targets)
    }
}
