//! Hardware tests: these need a Mac running macOS 26+ with a GPU that supports the denoiser.
#![cfg(target_os = "macos")]

use metalfx_wgpu::{DenoiserDescriptor, DenoiserParams, DenoiserTextures, Error, MetalFxDenoiser};

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::debugging(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("no Metal adapter");
    pollster::block_on(adapter.request_device(&Default::default())).expect("no device")
}

struct Frame {
    color: wgpu::Texture,
    depth: wgpu::Texture,
    motion: wgpu::Texture,
    diffuse_albedo: wgpu::Texture,
    specular_albedo: wgpu::Texture,
    normal: wgpu::Texture,
    roughness: wgpu::Texture,
    output: wgpu::Texture,
}

impl Frame {
    fn textures(&self) -> DenoiserTextures<'_> {
        DenoiserTextures {
            color: &self.color,
            depth: &self.depth,
            motion: &self.motion,
            diffuse_albedo: &self.diffuse_albedo,
            specular_albedo: &self.specular_albedo,
            normal: &self.normal,
            roughness: &self.roughness,
            output: &self.output,
        }
    }
}

const COLOR: [f64; 4] = [0.5, 0.25, 0.125, 1.0];

/// Creates every texture and clears it to a constant through wgpu, so wgpu considers it
/// initialized. The output starts out magenta, which tells us whether MetalFX overwrote it.
fn frame(device: &wgpu::Device, queue: &wgpu::Queue, desc: &DenoiserDescriptor) -> Frame {
    let input = |format, label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: desc.input_size[0],
                height: desc.input_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    };
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d {
            width: desc.output_size[0],
            height: desc.output_size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.output_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let f = Frame {
        color: input(desc.color_format, "color"),
        depth: input(desc.depth_format, "depth"),
        motion: input(desc.motion_format, "motion"),
        diffuse_albedo: input(desc.diffuse_albedo_format, "diffuse_albedo"),
        specular_albedo: input(desc.specular_albedo_format, "specular_albedo"),
        normal: input(desc.normal_format, "normal"),
        roughness: input(desc.roughness_format, "roughness"),
        output,
    };

    let mut encoder = device.create_command_encoder(&Default::default());
    for (texture, [r, g, b, a]) in [
        (&f.color, COLOR),
        (&f.depth, [0.5, 0.0, 0.0, 0.0]),
        (&f.motion, [0.0; 4]),
        (&f.diffuse_albedo, [0.8, 0.8, 0.8, 1.0]),
        (&f.specular_albedo, [0.04, 0.04, 0.04, 1.0]),
        (&f.normal, [0.0, 0.0, 1.0, 0.0]),
        (&f.roughness, [0.5, 0.0, 0.0, 0.0]),
        (&f.output, [1.0, 0.0, 1.0, 1.0]),
    ] {
        let view = texture.create_view(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    queue.submit([encoder.finish()]);
    f
}

/// Reads back an Rgba16Float texture as f32 RGBA texels.
fn read_rgba16f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<[f32; 4]> {
    let (w, h) = (texture.width(), texture.height());
    let bytes_per_row = (w * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.get_mapped_range(..).unwrap();
    let mut texels = Vec::with_capacity((w * h) as usize);
    for row in data.chunks(bytes_per_row as usize) {
        for px in row[..(w * 8) as usize].chunks(8) {
            let c = |i: usize| half::f16::from_le_bytes([px[i], px[i + 1]]).to_f32();
            texels.push([c(0), c(2), c(4), c(6)]);
        }
    }
    texels
}

#[test]
fn reports_support() {
    let (device, _) = device();
    let support = MetalFxDenoiser::support(&device).expect("denoiser unsupported");
    assert!(
        support.min_scale <= 1.0 && support.max_scale >= 2.0,
        "{support:?}"
    );
}

#[test]
fn reports_required_usages() {
    let (device, _) = device();
    let denoiser =
        MetalFxDenoiser::new(&device, DenoiserDescriptor::solari([256, 256], [256, 256])).unwrap();
    let u = denoiser.required_usages();
    assert_eq!(u.color, wgpu::TextureUsages::TEXTURE_BINDING);
    assert_eq!(
        u.output,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
    );
}

#[test]
fn rejects_out_of_range_scale() {
    let (device, _) = device();
    let result = MetalFxDenoiser::new(
        &device,
        DenoiserDescriptor::solari([256, 256], [1024, 1024]),
    );
    assert!(
        matches!(result, Err(Error::UnsupportedScale { .. })),
        "{:?}",
        result.err()
    );
}

#[test]
fn matrices_round_trip() {
    let (device, _) = device();
    let mut denoiser =
        MetalFxDenoiser::new(&device, DenoiserDescriptor::solari([256, 256], [256, 256])).unwrap();
    assert_eq!(
        denoiser.world_to_view(),
        [[0.0; 4]; 4],
        "MetalFX default should be zero"
    );

    let a: [[f32; 4]; 4] =
        std::array::from_fn(|c| std::array::from_fn(|r| (c * 4 + r) as f32 + 0.5));
    let b: [[f32; 4]; 4] =
        std::array::from_fn(|c| std::array::from_fn(|r| -((c * 4 + r) as f32) * 3.25));
    denoiser.set_world_to_view(a);
    denoiser.set_view_to_clip(b);
    assert_eq!(denoiser.world_to_view(), a);
    assert_eq!(denoiser.view_to_clip(), b);
}

#[test]
fn rejects_mismatched_texture() {
    let (device, queue) = device();
    let desc = DenoiserDescriptor::solari([128, 128], [128, 128]);
    let mut denoiser = MetalFxDenoiser::new(&device, desc).unwrap();
    let wrong = frame(
        &device,
        &queue,
        &DenoiserDescriptor::solari([64, 64], [128, 128]),
    );
    let result = denoiser.encode(&device, wrong.textures(), &DenoiserParams::default());
    assert!(
        matches!(
            result,
            Err(Error::TextureMismatch {
                texture: "color",
                ..
            })
        ),
        "{:?}",
        result.err()
    );
}

/// A constant-color input should come back (almost) unchanged, at both native and upscaled sizes.
#[test]
fn encodes_constant_image() {
    let (device, queue) = device();
    for (input, output) in [([256, 256], [256, 256]), ([128, 128], [256, 256])] {
        let desc = DenoiserDescriptor {
            requires_synchronous_initialization: true,
            auto_exposure: false,
            ..DenoiserDescriptor::solari(input, output)
        };
        let mut denoiser = MetalFxDenoiser::new(&device, desc).unwrap();
        let f = frame(&device, &queue, &desc);
        for i in 0..8 {
            let params = DenoiserParams {
                reset: i == 0,
                ..Default::default()
            };
            let cmd = denoiser.encode(&device, f.textures(), &params).unwrap();
            queue.submit([cmd]);
        }
        let texels = read_rgba16f(&device, &queue, &f.output);
        let center = texels[texels.len() / 2 + output[0] as usize / 2];
        println!("{input:?} -> {output:?}: center texel {center:?}");
        let max_err = texels
            .iter()
            .flat_map(|t| (0..3).map(move |c| (t[c] - COLOR[c] as f32).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.02,
            "{input:?} -> {output:?}: max error {max_err}, center {center:?}"
        );
    }
}
