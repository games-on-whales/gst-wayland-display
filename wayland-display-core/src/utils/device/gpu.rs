use crate::utils::device::PCIVendor;
use smithay::backend::drm::DrmNode;
use std::error::Error;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct GPUDevice {
    pci_vendor: PCIVendor,
    device_name: String,
}
impl GPUDevice {
    pub fn pci_vendor(&self) -> &PCIVendor {
        &self.pci_vendor
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}
impl TryFrom<DrmNode> for GPUDevice {
    type Error = Box<dyn Error>;
    fn try_from(drm_node: DrmNode) -> Result<Self, Self::Error> {
        get_gpu_device(drm_node.dev_path().unwrap().to_str().unwrap())
    }
}
impl std::fmt::Display for GPUDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPUDevice {{ pci_vendor: {}, device_name: {} }}",
            self.pci_vendor, self.device_name
        )
    }
}

pub fn get_gpu_device(path: &str) -> Result<GPUDevice, Box<dyn Error>> {
    let card = get_card_from_render_node(path)?;
    let vendor_str = fs::read_to_string(format!("/sys/class/drm/{}/device/vendor", card))?;
    let vendor_str = vendor_str.trim_start_matches("0x").trim_end_matches('\n');
    let vendor = u32::from_str_radix(&vendor_str, 16)?;

    let device_id = fs::read_to_string(format!("/sys/class/drm/{}/device/device", card))?;
    let device_id = device_id.trim_start_matches("0x").trim_end_matches('\n');

    // Look up in hwdata PCI database
    let device_name = match fs::read_to_string("/usr/share/hwdata/pci.ids") {
        Ok(pci_ids) => parse_pci_ids(&pci_ids, vendor_str, device_id).unwrap_or("".to_owned()),
        Err(e) => {
            tracing::warn!("Failed to read /usr/share/hwdata/pci.ids: {}", e);
            "".to_owned()
        }
    };

    Ok(GPUDevice {
        pci_vendor: PCIVendor::try_from(vendor)?,
        device_name,
    })
}

fn parse_pci_ids(pci_data: &str, vendor_id: &str, device_id: &str) -> Option<String> {
    let mut current_vendor = String::new();
    let vendor_id = vendor_id.to_lowercase();
    let device_id = device_id.to_lowercase();

    for line in pci_data.lines() {
        // Skip comments and empty lines
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Check for vendor lines (no leading whitespace)
        if !line.starts_with(['\t', ' ']) {
            let mut parts = line.splitn(2, ' ');
            if let (Some(vendor), Some(_)) = (parts.next(), parts.next()) {
                current_vendor = vendor.to_lowercase();
            }
            continue;
        }

        // Check for device lines (leading whitespace)
        let line = line.trim_start();
        let mut parts = line.splitn(2, ' ');
        if let (Some(dev_id), Some(desc)) = (parts.next(), parts.next()) {
            if dev_id.to_lowercase() == device_id && current_vendor == vendor_id {
                return Some(desc.trim().to_owned());
            }
        }
    }

    None
}

fn get_card_from_render_node(render_path: &str) -> std::io::Result<String> {
    // Get the device's sysfs path
    let metadata = fs::metadata(render_path)?;
    let rdev = metadata.rdev();
    let major = gnu_dev_major(rdev);
    let minor = gnu_dev_minor(rdev);

    // The sysfs path for the device
    let sys_path = format!("/sys/dev/char/{}:{}", major, minor);

    // Read the device symlink to get the actual device
    let device_link = PathBuf::from(&sys_path).join("device");
    let device_real = fs::canonicalize(device_link)?;

    // Now find card* entries under the same device
    let drm_path = device_real.join("drm");
    for entry in fs::read_dir(drm_path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("card") && !name.contains('-') {
            return Ok(name);
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "No card found",
    ))
}

fn gnu_dev_major(dev: u64) -> u32 {
    let mut major = 0;
    major |= ((dev >> 8) & 0xfff) as u32;
    major |= ((dev >> 32) & 0xfffff000) as u32;
    major
}

fn gnu_dev_minor(dev: u64) -> u32 {
    let mut minor = 0;
    minor |= (dev & 0xff) as u32;
    minor |= ((dev >> 12) & 0xffffff00) as u32;
    minor
}

/// Get PCI bus ID from NVIDIA render-node
/// 
/// # Arguments
/// * `render_path` - DRM render node path, e.g., "/dev/dri/renderD128"
/// 
/// # Returns
/// Returns PCI bus ID in format "00000000:01:00.0"
pub fn get_pci_bus_id_from_render_node(render_path: &str) -> Result<String, Box<dyn Error>> {
    let card = get_card_from_render_node(render_path)?;
    
    // Read PCI information from /sys/class/drm/cardX/device/uevent
    let uevent_path = format!("/sys/class/drm/{}/device/uevent", card);
    let uevent_content = fs::read_to_string(uevent_path)?;
    
    // Find PCI_SLOT_NAME in format "0000:01:00.0"
    for line in uevent_content.lines() {
        if line.starts_with("PCI_SLOT_NAME=") {
            let pci_slot = line.strip_prefix("PCI_SLOT_NAME=").unwrap();
            // Convert to nvidia-smi format: "00000000:01:00.0"
            let parts: Vec<&str> = pci_slot.split(':').collect();
            if parts.len() == 3 {
                return Ok(format!("00000000:{}:{}", parts[1], parts[2]));
            }
        }
    }
    
    Err("Failed to find PCI bus ID from render node".into())
}

/// Get CUDA device ID from NVIDIA render-node (without using nvidia-smi)
/// 
/// # Arguments
/// * `render_path` - DRM render node path, e.g., "/dev/dri/renderD128"
/// 
/// # Returns
/// Returns CUDA device ID (typically 0, 1, 2, ...)
/// 
/// # Method
/// 1. Get PCI bus ID from render-node (format: 0000:01:00.0)
/// 2. Iterate through /proc/driver/nvidia/gpus/ directory to find matching PCI bus ID
/// 3. CUDA device ID is assigned in alphabetical order of PCI bus IDs (this is CUDA's default behavior)
/// 
/// # Note
/// This method does not depend on nvidia-smi, but requires NVIDIA driver to be loaded
/// and /proc/driver/nvidia/gpus/ directory to be accessible
pub fn get_cuda_device_id_from_render_node(render_path: &str) -> Result<i32, Box<dyn Error>> {
    // Get PCI bus ID from render-node (format: 0000:01:00.0)
    let card = get_card_from_render_node(render_path)?;
    let uevent_path = format!("/sys/class/drm/{}/device/uevent", card);
    let uevent_content = fs::read_to_string(uevent_path)?;
    
    let mut pci_slot_name: Option<&str> = None;
    for line in uevent_content.lines() {
        if line.starts_with("PCI_SLOT_NAME=") {
            pci_slot_name = Some(line.strip_prefix("PCI_SLOT_NAME=").unwrap());
            break;
        }
    }
    
    let pci_slot = pci_slot_name.ok_or("Failed to find PCI_SLOT_NAME from render node")?;
    
    // Read /proc/driver/nvidia/gpus/ directory
    // Each subdirectory name in this directory is the PCI bus ID of the corresponding GPU
    let nvidia_gpus_dir = "/proc/driver/nvidia/gpus";
    let entries: Vec<_> = fs::read_dir(nvidia_gpus_dir)
        .map_err(|e| format!("Failed to read {}: {}. Make sure NVIDIA driver is loaded and accessible.", nvidia_gpus_dir, e))?
        .collect::<Result<Vec<_>, _>>()?;
    
    if entries.is_empty() {
        return Err("No NVIDIA GPUs found in /proc/driver/nvidia/gpus/".into());
    }
    
    // Collect PCI bus IDs of all GPUs
    let mut gpus: Vec<String> = Vec::new();
    for entry in &entries {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        // Directory name is the PCI bus ID (format: 0000:01:00.0)
        gpus.push(dir_name);
    }
    
    // Sort by PCI bus ID
    // CUDA device ID is assigned in alphabetical order of PCI bus IDs (this is CUDA's default behavior)
    gpus.sort();
    
    // Find matching PCI bus ID, the index is the CUDA device ID
    for (cuda_id, pci_id) in gpus.iter().enumerate() {
        if pci_id == pci_slot {
            return Ok(cuda_id as i32);
        }
    }
    
    Err(format!("No CUDA device found for PCI bus ID: {}. Available GPUs: {:?}", 
        pci_slot, gpus).into())
}

/// Get CUDA device ID from DrmNode
pub fn get_cuda_device_id_from_drm_node(drm_node: DrmNode) -> Result<i32, Box<dyn Error>> {
    let render_path = drm_node.dev_path()
        .ok_or("Failed to get device path from DrmNode")?;
    get_cuda_device_id_from_render_node(render_path.to_str().unwrap())
}
