mod model;
mod training;
mod data;

use burn::tensor::{Device, DeviceKind, DeviceIndex};

fn select_device() -> Device {
    #[cfg(feature = "vulkan")]
    return Device::vulkan(DeviceKind::DefaultDevice);
    #[cfg(feature = "metal")]
    return Device::metal(DeviceKind::DefaultDevice);
    #[cfg(feature = "wgpu")]
    return Device::wgpu(DeviceKind::DefaultDevice);
    #[cfg(feature = "cuda")]
    return Device::cuda(DeviceIndex::Default);
    #[cfg(feature = "rocm")]
    return Device::rocm(DeviceIndex::Default);

    // fallback, has the widest general support
    Device::flex()
}

fn main() {

    let device = select_device();
    training::run(device);
}
