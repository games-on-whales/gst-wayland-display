use crate::utils::device::PCIVendor;
use smithay::backend::drm::DrmNode;
use smithay::backend::vulkan::version::Version;
use smithay::backend::vulkan::{Instance, PhysicalDevice};

#[derive(Debug, Clone, PartialEq)]
pub struct GPUDevice {
    drm_node: DrmNode,
    pci_vendor: PCIVendor,
    device_name: String,
}
impl GPUDevice {
    pub fn drm_node(&self) -> &DrmNode {
        &self.drm_node
    }

    pub fn pci_vendor(&self) -> &PCIVendor {
        &self.pci_vendor
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}
impl TryFrom<DrmNode> for GPUDevice {
    type Error = Box<dyn std::error::Error>;
    fn try_from(drm_node: DrmNode) -> Result<Self, Self::Error> {
        let devices = enumerate_gpu_devices()?;
        if let Some(device) = devices.iter().find(|d| d.drm_node == drm_node) {
            Ok(device.clone())
        } else {
            Err("No GPU device for given DRM node".into())
        }
    }
}
impl std::fmt::Display for GPUDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPUDevice {{ drm_node: {}, pci_vendor: {}, device_name: {} }}",
            self.drm_node, self.pci_vendor, self.device_name
        )
    }
}

pub fn enumerate_gpu_devices() -> Result<Vec<GPUDevice>, Box<dyn std::error::Error>> {
    let mut devices = Vec::new();

    let instance = Instance::new(Version::VERSION_1_1, None)?;

    for p_dev in PhysicalDevice::enumerate(&instance)? {
        // Add only devices that support DrmNode (filters out software devices)
        let drm_node: DrmNode = if let Ok(render_node) = p_dev.render_node()
            && let Some(render_node) = render_node
        {
            render_node
        } else if let Ok(primary_node) = p_dev.primary_node()
            && let Some(primary_node) = primary_node
        {
            primary_node
        } else {
            continue;
        };

        let properties = p_dev.properties();
        let pci_vendor = PCIVendor::try_from(properties.vendor_id);

        // array of c_char's (i8) needs conversion to String
        let device_name = properties
            .device_name
            .as_slice()
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect::<String>();

        devices.push(GPUDevice {
            drm_node,
            pci_vendor: pci_vendor.unwrap_or(PCIVendor::Unknown),
            device_name,
        });
    }

    Ok(devices)
}
