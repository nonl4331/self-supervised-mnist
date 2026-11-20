use burn::{
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    prelude::*,
    train::ClassificationOutput,
};

use crate::data;

#[derive(Module, Debug)]
pub struct Model {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Model {
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
    pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch_size, _, _] = input.dims();
        assert_shape!(input, [_, data::IMAGE_X, data::IMAGE_Y]);
        let x = input.reshape([batch_size, data::INPUT_SIZE]);
        let x = self.fc1.forward(x);
        let x = nn::Relu::new().forward(x);
        let x = self.fc2.forward(x);
        nn::Relu::new().forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Model2 {
    pub backbone: Model2Backbone,
    linear_probe: nn::Linear,
}

impl Model2 {
    pub fn new(device: &Device, backbone: Model2Backbone) -> Self {
        Self {
            backbone,
            linear_probe: nn::LinearConfig::new(10, 10).init(device),
        }
    }
    pub fn foward_classification(
        &self,
        input: Tensor<3>,
        target: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let out = self.linear_probe.forward(self.backbone.forward(input));
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out.clone(), target.clone());
        ClassificationOutput::new(loss, out, target)
    }
}
