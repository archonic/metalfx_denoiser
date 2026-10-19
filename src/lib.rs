//! MetalFX temporal denoised scaler (`MTLFXTemporalDenoisedScaler`) for wgpu's Metal backend.
//!
//! The denoiser works on textures created by wgpu and encodes into a command buffer of its own,
//! which you submit like any other [`wgpu::CommandBuffer`]. MetalFX work is never recorded into an
//! encoder that also carries wgpu commands, because mixing `as_hal_mut` with regular encoding on
//! one encoder isn't supported.
//!
//! # Texture initialization
//! wgpu doesn't know that MetalFX reads the inputs and writes the output. If wgpu's first use of
//! the output texture is a read, wgpu lazily clears it first, which would wipe what MetalFX wrote.
//! Write to or clear every texture through wgpu at least once (Bevy's view targets already are).

#[cfg(target_os = "macos")]
mod metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod simd;

#[cfg(target_os = "macos")]
pub use metal::MetalFxDenoiser;

/// Errors from creating the denoiser or encoding with it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the wgpu device is not using the Metal backend")]
    NotMetal,
    #[error(
        "MTLFXTemporalDenoisedScaler is not supported on this device or OS (requires macOS 26+)"
    )]
    Unsupported,
    #[error("texture format {0:?} is not supported by this crate")]
    UnsupportedFormat(wgpu::TextureFormat),
    #[error("input content scale {scale} is outside the supported range {min}..={max}")]
    UnsupportedScale { scale: f32, min: f32, max: f32 },
    #[error("MetalFX failed to create the denoised scaler")]
    CreationFailed,
    #[error("{texture} texture: {reason}")]
    TextureMismatch {
        texture: &'static str,
        reason: String,
    },
}

/// What the device supports, as reported by MetalFX.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Support {
    /// Smallest supported `output / input` scale factor.
    pub min_scale: f32,
    /// Largest supported `output / input` scale factor.
    pub max_scale: f32,
}

/// Configuration fixed at creation time. Recreate the denoiser to change any of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenoiserDescriptor {
    /// Render resolution: size of every input texture.
    pub input_size: [u32; 2],
    /// Size of the output texture.
    pub output_size: [u32; 2],
    pub color_format: wgpu::TextureFormat,
    pub output_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub motion_format: wgpu::TextureFormat,
    pub diffuse_albedo_format: wgpu::TextureFormat,
    pub specular_albedo_format: wgpu::TextureFormat,
    pub normal_format: wgpu::TextureFormat,
    pub roughness_format: wgpu::TextureFormat,
    /// Let MetalFX compute exposure each frame. If false, MetalFX reads no exposure texture yet
    /// (one isn't wired up), so exposure is effectively 1.0.
    pub auto_exposure: bool,
    /// Compile the fast pipelines while creating the scaler instead of in the background.
    pub requires_synchronous_initialization: bool,
}

impl DenoiserDescriptor {
    /// Formats that match bevy_solari's guide buffers and Bevy's HDR view target.
    pub fn solari(input_size: [u32; 2], output_size: [u32; 2]) -> Self {
        use wgpu::TextureFormat as F;
        Self {
            input_size,
            output_size,
            color_format: F::Rgba16Float,
            output_format: F::Rgba16Float,
            depth_format: F::R32Float,
            motion_format: F::Rg16Float,
            diffuse_albedo_format: F::Rgba8Unorm,
            specular_albedo_format: F::Rgba8Unorm,
            normal_format: F::Rgba16Float,
            roughness_format: F::R16Float,
            auto_exposure: true,
            requires_synchronous_initialization: false,
        }
    }
}

/// Minimum wgpu usages each texture must be created with, derived from the scaler's
/// `*TextureUsage` properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredUsages {
    pub color: wgpu::TextureUsages,
    pub depth: wgpu::TextureUsages,
    pub motion: wgpu::TextureUsages,
    pub diffuse_albedo: wgpu::TextureUsages,
    pub specular_albedo: wgpu::TextureUsages,
    pub normal: wgpu::TextureUsages,
    pub roughness: wgpu::TextureUsages,
    pub output: wgpu::TextureUsages,
}

/// Textures for one frame.
///
/// Inputs must be at least `input_size`. If one is larger, its top-left `input_size` region is
/// used: it is copied into an exact-size texture first, because MetalFX itself rejects any size
/// mismatch. This matches how Bevy renders into part of a full-size view target under a resolution
/// override. The output must be exactly `output_size`.
#[derive(Clone, Copy)]
pub struct DenoiserTextures<'a> {
    /// Noisy HDR color.
    pub color: &'a wgpu::Texture,
    pub depth: &'a wgpu::Texture,
    pub motion: &'a wgpu::Texture,
    pub diffuse_albedo: &'a wgpu::Texture,
    pub specular_albedo: &'a wgpu::Texture,
    pub normal: &'a wgpu::Texture,
    pub roughness: &'a wgpu::Texture,
    pub output: &'a wgpu::Texture,
}

/// Per-frame parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenoiserParams {
    /// Sub-pixel jitter of this frame, in pixels.
    pub jitter_offset: [f32; 2],
    /// Multiplier that turns the stored motion vectors into pixel offsets pointing to the previous
    /// frame.
    pub motion_vector_scale: [f32; 2],
    /// Discard history, e.g. on a camera cut.
    pub reset: bool,
    /// Depth uses 0.0 for the far plane (Bevy's reverse-Z).
    pub depth_reversed: bool,
    pub pre_exposure: f32,
    /// Column-major world-to-view matrix.
    pub world_to_view: [[f32; 4]; 4],
    /// Column-major view-to-clip (projection) matrix.
    pub view_to_clip: [[f32; 4]; 4],
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl Default for DenoiserParams {
    fn default() -> Self {
        Self {
            jitter_offset: [0.0; 2],
            motion_vector_scale: [1.0; 2],
            reset: false,
            depth_reversed: true,
            pre_exposure: 1.0,
            world_to_view: IDENTITY,
            view_to_clip: IDENTITY,
        }
    }
}
