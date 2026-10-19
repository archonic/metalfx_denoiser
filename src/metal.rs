use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject, ProtocolObject, Sel};
use objc2::{Message, sel};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLOrigin,
    MTLPixelFormat, MTLResource, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use objc2_metal_fx::{
    MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use wgpu::hal::api::Metal;

#[cfg(target_arch = "aarch64")]
use crate::simd;
use crate::{DenoiserDescriptor, DenoiserParams, DenoiserTextures, Error, RequiredUsages, Support};

/// A `MTLFXTemporalDenoisedScaler` bound to one wgpu device and one fixed configuration.
pub struct MetalFxDenoiser {
    scaler: Retained<ProtocolObject<dyn MTLFXTemporalDenoisedScaler>>,
    descriptor: DenoiserDescriptor,
    /// GPU time of the last completed encode, as `f64` seconds bits. `None` if timing is off.
    gpu_time: Option<Arc<AtomicU64>>,
    /// Exact-size copies of oversized inputs, indexed like [`INPUT_NAMES`].
    scratch: [Option<Retained<ProtocolObject<dyn MTLTexture>>>; 7],
}

/// Inputs in the order the scaler's texture properties are handled.
const INPUT_NAMES: [&str; 7] = [
    "color",
    "depth",
    "motion",
    "diffuse albedo",
    "specular albedo",
    "normal",
    "roughness",
];

// SAFETY: Metal objects can be used from any thread. Every call that changes the scaler's state
// takes `&mut self`, so no two threads can configure or encode with it at the same time.
unsafe impl Send for MetalFxDenoiser {}
unsafe impl Sync for MetalFxDenoiser {}

fn with_raw_device<R>(
    device: &wgpu::Device,
    f: impl FnOnce(&ProtocolObject<dyn MTLDevice>) -> R,
) -> Result<R, Error> {
    // SAFETY: the raw device is only borrowed for the duration of `f` and never destroyed.
    let hal_device = unsafe { device.as_hal::<Metal>() }.ok_or(Error::NotMetal)?;
    Ok(f(hal_device.raw_device()))
}

/// `(setter, getter)` selectors of the matrix properties.
fn world_to_view_sels() -> (Sel, Sel) {
    (sel!(setWorldToViewMatrix:), sel!(worldToViewMatrix))
}
fn view_to_clip_sels() -> (Sel, Sel) {
    (sel!(setViewToClipMatrix:), sel!(viewToClipMatrix))
}

fn descriptor_class_available() -> bool {
    // Only present on macOS 26+; using the class on an older OS would panic.
    AnyClass::get(c"MTLFXTemporalDenoisedScalerDescriptor").is_some()
}

impl MetalFxDenoiser {
    /// Returns `None` if the device isn't Metal or doesn't support the denoised scaler.
    pub fn support(device: &wgpu::Device) -> Option<Support> {
        if !descriptor_class_available() {
            return None;
        }
        with_raw_device(device, |raw| unsafe {
            MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(raw).then(|| Support {
                min_scale:
                    MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(
                        raw,
                    ),
                max_scale:
                    MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(
                        raw,
                    ),
            })
        })
        .ok()
        .flatten()
    }

    pub fn new(device: &wgpu::Device, descriptor: DenoiserDescriptor) -> Result<Self, Error> {
        // MetalFX autoreleases objects. Bevy's render thread has no autorelease pool, so drain
        // our own or they pile up until the thread exits.
        autoreleasepool(|_| Self::new_inner(device, descriptor))
    }

    fn new_inner(device: &wgpu::Device, descriptor: DenoiserDescriptor) -> Result<Self, Error> {
        let support = Self::support(device).ok_or(Error::Unsupported)?;
        let d = &descriptor;
        for scale in [
            d.output_size[0] as f32 / d.input_size[0] as f32,
            d.output_size[1] as f32 / d.input_size[1] as f32,
        ] {
            if !(support.min_scale..=support.max_scale).contains(&scale) {
                return Err(Error::UnsupportedScale {
                    scale,
                    min: support.min_scale,
                    max: support.max_scale,
                });
            }
        }

        let desc = unsafe { MTLFXTemporalDenoisedScalerDescriptor::new() };
        unsafe {
            desc.setColorTextureFormat(map_format(d.color_format)?);
            desc.setOutputTextureFormat(map_format(d.output_format)?);
            desc.setDepthTextureFormat(map_format(d.depth_format)?);
            desc.setMotionTextureFormat(map_format(d.motion_format)?);
            desc.setDiffuseAlbedoTextureFormat(map_format(d.diffuse_albedo_format)?);
            desc.setSpecularAlbedoTextureFormat(map_format(d.specular_albedo_format)?);
            desc.setNormalTextureFormat(map_format(d.normal_format)?);
            desc.setRoughnessTextureFormat(map_format(d.roughness_format)?);
            desc.setInputWidth(d.input_size[0] as usize);
            desc.setInputHeight(d.input_size[1] as usize);
            desc.setOutputWidth(d.output_size[0] as usize);
            desc.setOutputHeight(d.output_size[1] as usize);
            desc.setAutoExposureEnabled(d.auto_exposure);
            desc.setRequiresSynchronousInitialization(d.requires_synchronous_initialization);
        }

        let scaler = with_raw_device(device, |raw| unsafe {
            desc.newTemporalDenoisedScalerWithDevice(raw)
        })?
        .ok_or(Error::CreationFailed)?;

        // The matrix setters go through hand-written ABI glue (see `simd.rs`). Refuse to run on
        // targets without it, or if the runtime signature isn't the one the glue expects.
        #[cfg(target_arch = "aarch64")]
        let matrices_ok = simd::has_matrix_methods(
            as_object(&scaler),
            &[world_to_view_sels(), view_to_clip_sels()],
        );
        #[cfg(not(target_arch = "aarch64"))]
        let matrices_ok = false;
        if !matrices_ok {
            return Err(Error::Unsupported);
        }

        Ok(Self {
            scaler,
            descriptor,
            gpu_time: None,
            scratch: Default::default(),
        })
    }

    pub fn descriptor(&self) -> &DenoiserDescriptor {
        &self.descriptor
    }

    pub fn required_usages(&self) -> RequiredUsages {
        let s = &self.scaler;
        unsafe {
            RequiredUsages {
                color: map_usage(s.colorTextureUsage()),
                depth: map_usage(s.depthTextureUsage()),
                motion: map_usage(s.motionTextureUsage()),
                diffuse_albedo: map_usage(s.diffuseAlbedoTextureUsage()),
                specular_albedo: map_usage(s.specularAlbedoTextureUsage()),
                normal: map_usage(s.normalTextureUsage()),
                roughness: map_usage(s.roughnessTextureUsage()),
                output: map_usage(s.outputTextureUsage()),
            }
        }
    }

    /// Encode one frame into a new command buffer. Submit it after the work that writes the inputs
    /// and before the work that reads the output.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        textures: DenoiserTextures<'_>,
        params: &DenoiserParams,
    ) -> Result<wgpu::CommandBuffer, Error> {
        autoreleasepool(|_| self.encode_inner(device, textures, params))
    }

    fn encode_inner(
        &mut self,
        device: &wgpu::Device,
        textures: DenoiserTextures<'_>,
        params: &DenoiserParams,
    ) -> Result<wgpu::CommandBuffer, Error> {
        let s = &self.scaler;
        let input = self.descriptor.input_size;
        let output = self.descriptor.output_size;

        // wgpu can't hold several `as_hal` guards at once (it panics on the recursive snatch lock),
        // so retain each raw texture and release its guard right away.
        let hal = |t: &wgpu::Texture| {
            let guard = unsafe { t.as_hal::<Metal>() }.ok_or(Error::NotMetal)?;
            Ok::<_, Error>(guard.raw_handle().retain())
        };
        let (formats, usages) = unsafe {
            (
                [
                    s.colorTextureFormat(),
                    s.depthTextureFormat(),
                    s.motionTextureFormat(),
                    s.diffuseAlbedoTextureFormat(),
                    s.specularAlbedoTextureFormat(),
                    s.normalTextureFormat(),
                    s.roughnessTextureFormat(),
                ],
                [
                    s.colorTextureUsage(),
                    s.depthTextureUsage(),
                    s.motionTextureUsage(),
                    s.diffuseAlbedoTextureUsage(),
                    s.specularAlbedoTextureUsage(),
                    s.normalTextureUsage(),
                    s.roughnessTextureUsage(),
                ],
            )
        };
        let sources = [
            textures.color,
            textures.depth,
            textures.motion,
            textures.diffuse_albedo,
            textures.specular_albedo,
            textures.normal,
            textures.roughness,
        ];

        // Inputs larger than `input_size` (e.g. Bevy's view target under a resolution override)
        // are cropped to their top-left region into scratch textures: MetalFX requires exact sizes.
        let mut inputs = Vec::with_capacity(INPUT_NAMES.len());
        let mut crops = Vec::new();
        for (i, source) in sources.into_iter().enumerate() {
            let raw = hal(source)?;
            let name = INPUT_NAMES[i];
            validate(name, &raw, formats[i], usages[i])?;
            let size = [raw.width() as u32, raw.height() as u32];
            if size == input {
                inputs.push(raw);
            } else if size[0] >= input[0] && size[1] >= input[1] {
                let scratch = self.scratch_texture(i, &raw, formats[i], usages[i])?;
                crops.push((raw, scratch.clone()));
                inputs.push(scratch);
            } else {
                return Err(Error::TextureMismatch {
                    texture: name,
                    reason: format!("size is {size:?}, smaller than the input size {input:?}"),
                });
            }
        }
        let out = hal(textures.output)?;
        let s = &self.scaler;
        unsafe {
            validate(
                "output",
                &out,
                s.outputTextureFormat(),
                s.outputTextureUsage(),
            )?
        };
        let out_size = [out.width() as u32, out.height() as u32];
        if out_size != output {
            return Err(Error::TextureMismatch {
                texture: "output",
                reason: format!("size is {out_size:?}, expected {output:?}"),
            });
        }
        if out.storageMode() != MTLStorageMode::Private {
            return Err(Error::TextureMismatch {
                texture: "output",
                reason: "MetalFX requires private storage mode".into(),
            });
        }

        unsafe {
            s.setColorTexture(Some(&inputs[0]));
            s.setDepthTexture(Some(&inputs[1]));
            s.setMotionTexture(Some(&inputs[2]));
            s.setDiffuseAlbedoTexture(Some(&inputs[3]));
            s.setSpecularAlbedoTexture(Some(&inputs[4]));
            s.setNormalTexture(Some(&inputs[5]));
            s.setRoughnessTexture(Some(&inputs[6]));
            s.setOutputTexture(Some(&out));

            s.setJitterOffsetX(params.jitter_offset[0]);
            s.setJitterOffsetY(params.jitter_offset[1]);
            s.setMotionVectorScaleX(params.motion_vector_scale[0]);
            s.setMotionVectorScaleY(params.motion_vector_scale[1]);
            s.setShouldResetHistory(params.reset);
            s.setDepthReversed(params.depth_reversed);
            s.setPreExposure(params.pre_exposure);
        }
        send_matrix(s, world_to_view_sels().0, params.world_to_view);
        send_matrix(s, view_to_clip_sels().0, params.view_to_clip);

        // wgpu-hal records into command buffers with unretained references, and wgpu doesn't know
        // which textures this work uses. Keep every one alive until the GPU is done with it.
        let mut keep_alive: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = inputs.clone();
        keep_alive.push(out.clone());
        keep_alive.extend(crops.iter().map(|(source, _)| source.clone()));

        // A dedicated encoder that only ever carries the MetalFX work.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("metalfx_denoiser"),
        });
        let gpu_time = self.gpu_time.clone();
        let encoded = unsafe {
            encoder.as_hal_mut::<Metal, _, _>(|hal_encoder| {
                let command_buffer = hal_encoder?.raw_command_buffer()?;
                if !crops.is_empty() {
                    let blit = command_buffer.blitCommandEncoder()?;
                    for (source, scratch) in &crops {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            source,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                            MTLSize { width: input[0] as usize, height: input[1] as usize, depth: 1 },
                            scratch,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                        );
                    }
                    blit.endEncoding();
                }
                s.encodeToCommandBuffer(command_buffer);

                let handler =
                    RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        let _ = &keep_alive;
                        if let Some(slot) = &gpu_time {
                            // The command buffer only holds the denoiser's work.
                            let cb = cb.as_ref();
                            slot.store(
                                (cb.GPUEndTime() - cb.GPUStartTime()).to_bits(),
                                Ordering::Relaxed,
                            );
                        }
                    });
                command_buffer.addCompletedHandler(RcBlock::as_ptr(&handler));
                Some(())
            })
        };

        // Drop the scaler's references so it doesn't keep textures alive across resizes.
        unsafe {
            s.setColorTexture(None);
            s.setDepthTexture(None);
            s.setMotionTexture(None);
            s.setDiffuseAlbedoTexture(None);
            s.setSpecularAlbedoTexture(None);
            s.setNormalTexture(None);
            s.setRoughnessTexture(None);
            s.setOutputTexture(None);
        }

        encoded.ok_or(Error::NotMetal)?;
        Ok(encoder.finish())
    }

    /// Exact-size texture that input `slot` is cropped into, (re)created to match `source`.
    fn scratch_texture(
        &mut self,
        slot: usize,
        source: &ProtocolObject<dyn MTLTexture>,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, Error> {
        let [width, height] = self.descriptor.input_size;
        if let Some(existing) = &self.scratch[slot] {
            return Ok(existing.clone());
        }
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(usage);
        descriptor.setStorageMode(MTLStorageMode::Private);
        let texture = source
            .device()
            .newTextureWithDescriptor(&descriptor)
            .ok_or(Error::TextureMismatch {
                texture: INPUT_NAMES[slot],
                reason: "failed to allocate a scratch texture for cropping".into(),
            })?;
        self.scratch[slot] = Some(texture.clone());
        Ok(texture)
    }

    /// Measure the GPU time of each encode (see [`Self::last_gpu_time`]).
    pub fn set_gpu_timing(&mut self, enabled: bool) {
        self.gpu_time = enabled.then(|| Arc::new(AtomicU64::new(f64::NAN.to_bits())));
    }

    /// GPU time of the most recently *completed* encode, if timing is on and one has completed.
    pub fn last_gpu_time(&self) -> Option<Duration> {
        let seconds = f64::from_bits(self.gpu_time.as_ref()?.load(Ordering::Relaxed));
        (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
    }

    /// The scaler's current `worldToViewMatrix` (column-major).
    pub fn world_to_view(&self) -> [[f32; 4]; 4] {
        receive_matrix(&self.scaler, world_to_view_sels().1)
    }

    /// The scaler's current `viewToClipMatrix` (column-major).
    pub fn view_to_clip(&self) -> [[f32; 4]; 4] {
        receive_matrix(&self.scaler, view_to_clip_sels().1)
    }

    pub fn set_world_to_view(&mut self, m: [[f32; 4]; 4]) {
        send_matrix(&self.scaler, world_to_view_sels().0, m);
    }

    pub fn set_view_to_clip(&mut self, m: [[f32; 4]; 4]) {
        send_matrix(&self.scaler, view_to_clip_sels().0, m);
    }
}

fn as_object(scaler: &ProtocolObject<dyn MTLFXTemporalDenoisedScaler>) -> &AnyObject {
    scaler.as_ref()
}

// `new` only succeeds if `has_matrix_methods` verified both selectors, so these are sound.
#[cfg(target_arch = "aarch64")]
fn send_matrix(
    scaler: &ProtocolObject<dyn MTLFXTemporalDenoisedScaler>,
    setter: Sel,
    m: [[f32; 4]; 4],
) {
    unsafe { simd::send_matrix(as_object(scaler), setter, m) }
}

#[cfg(target_arch = "aarch64")]
fn receive_matrix(
    scaler: &ProtocolObject<dyn MTLFXTemporalDenoisedScaler>,
    getter: Sel,
) -> [[f32; 4]; 4] {
    unsafe { simd::receive_matrix(as_object(scaler), getter) }
}

// Unreachable: `new` fails on targets without the ABI glue.
#[cfg(not(target_arch = "aarch64"))]
fn send_matrix(_: &ProtocolObject<dyn MTLFXTemporalDenoisedScaler>, _: Sel, _: [[f32; 4]; 4]) {
    unreachable!()
}

#[cfg(not(target_arch = "aarch64"))]
fn receive_matrix(_: &ProtocolObject<dyn MTLFXTemporalDenoisedScaler>, _: Sel) -> [[f32; 4]; 4] {
    unreachable!()
}

fn validate(
    name: &'static str,
    texture: &ProtocolObject<dyn MTLTexture>,
    format: MTLPixelFormat,
    usage: MTLTextureUsage,
) -> Result<(), Error> {
    let mismatch = |reason: String| {
        Err(Error::TextureMismatch {
            texture: name,
            reason,
        })
    };
    if texture.pixelFormat() != format {
        return mismatch(format!(
            "pixel format is {:?}, expected {format:?}",
            texture.pixelFormat()
        ));
    }
    if !texture.usage().contains(usage) {
        return mismatch(format!(
            "usage {:?} is missing {:?}",
            texture.usage(),
            usage
        ));
    }
    if texture.sampleCount() != 1 {
        return mismatch("must not be multisampled".into());
    }
    Ok(())
}

fn map_usage(usage: MTLTextureUsage) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    out.set(
        wgpu::TextureUsages::TEXTURE_BINDING,
        usage.contains(MTLTextureUsage::ShaderRead),
    );
    out.set(
        wgpu::TextureUsages::STORAGE_BINDING,
        usage.contains(MTLTextureUsage::ShaderWrite),
    );
    out.set(
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        usage.contains(MTLTextureUsage::RenderTarget),
    );
    out
}

fn map_format(format: wgpu::TextureFormat) -> Result<MTLPixelFormat, Error> {
    use wgpu::TextureFormat as F;
    Ok(match format {
        F::R8Unorm => MTLPixelFormat::R8Unorm,
        F::R16Unorm => MTLPixelFormat::R16Unorm,
        F::R16Float => MTLPixelFormat::R16Float,
        F::R32Float => MTLPixelFormat::R32Float,
        F::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        F::Rg16Snorm => MTLPixelFormat::RG16Snorm,
        F::Rg16Float => MTLPixelFormat::RG16Float,
        F::Rg32Float => MTLPixelFormat::RG32Float,
        F::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        F::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        F::Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        F::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        F::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        F::Rg11b10Ufloat => MTLPixelFormat::RG11B10Float,
        F::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        F::Rgba16Float => MTLPixelFormat::RGBA16Float,
        F::Rgba32Float => MTLPixelFormat::RGBA32Float,
        F::Depth32Float => MTLPixelFormat::Depth32Float,
        other => return Err(Error::UnsupportedFormat(other)),
    })
}
