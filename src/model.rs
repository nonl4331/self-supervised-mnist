use burn::{
    module::ModuleDisplay, nn::loss::{CrossEntropyLossConfig}, prelude::*, train::ClassificationOutput,
};

pub trait ModelForward {
    fn forward(&self, input: Tensor<3>) -> Tensor<2>;
}

use crate::data;

#[derive(Module, Debug)]
pub struct MnistMLPBaseline {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MnistMLPBaseline {
    pub fn new(device: &Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(data::INPUT_SIZE, 30).init(device),
            fc2: nn::LinearConfig::new(30, 10).init(device),
        }
    }
    // output: [batch_size, 10]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch_size, _, _] = input.dims();
        assert_shape!(input, [_, data::IMAGE_X, data::IMAGE_Y]);
        let x = input.reshape([batch_size, data::INPUT_SIZE]);
        let x = self.fc1.forward(x);
        let x = nn::Relu::new().forward(x);
        let x = self.fc2.forward(x);
        nn::Relu::new().forward(x)
    }
    pub fn foward_classification(
        &self,
        input: Tensor<3>,
        target: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let out = self.forward(input);
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out.clone(), target.clone());
        ClassificationOutput::new(loss, out, target)
    }
}

#[derive(Module, Debug)]
pub struct Model2Backbone {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Model2Backbone {
    pub fn new(device: &Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(data::INPUT_SIZE, 30).init(device),
            fc2: nn::LinearConfig::new(30, 10).init(device),
        }
    }
}

impl ModelForward for Model2Backbone {
    fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch_size, _, _] = input.dims();
        assert_shape!(input, [_, data::IMAGE_X, data::IMAGE_Y]);
        let x = input.reshape([batch_size, data::INPUT_SIZE]);
        let x = self.fc1.forward(x);
        let x = nn::Gelu::new().forward(x);
        let x = self.fc2.forward(x);
        nn::Gelu::new().forward(x)
    }
}

#[derive(Module, Debug)]
pub struct JointEmbeddingPredictor<M: Module + ModelForward + ModuleDisplay> {
    pub backbone: JointEmbeddingModel<M>,
    linear_probe: nn::Linear,
}

impl<M: Module + ModelForward + ModuleDisplay> JointEmbeddingPredictor<M> {
    pub fn new(device: &Device, backbone: JointEmbeddingModel<M>) -> Self {
        Self {
            backbone,
            linear_probe: nn::LinearConfig::new(20, 10).init(device),
        }
    }
    pub fn foward_classification(
        &self,
        input: Tensor<3>,
        target: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let (a, b) = self.backbone.out(input.clone(), input);
        let out = self.linear_probe.forward(Tensor::cat(vec![a, b], 1));
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out.clone(), target.clone());
        ClassificationOutput::new(loss, out, target)
    }
}


#[derive(Module, Debug)]
pub struct JointEmbeddingModel<M: Module + ModelForward> {
    modela: M,
    modelb: M,
}

impl<M: Module + ModelForward> JointEmbeddingModel<M> {
    pub fn new(modela: M, modelb: M) -> Self {
        Self { modela, modelb }
    }
    pub fn out(&self, input1: Tensor<3>, input2: Tensor<3>) -> (Tensor<2>, Tensor<2>) {
        (self.modela.forward(input1), self.modelb.forward(input2))
    }
}
