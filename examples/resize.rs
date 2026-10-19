//! Recreates the denoiser and its textures across resolution changes, as a Bevy app does when the
//! window is resized, and encodes a few frames at each size.
//!
//! ```sh
//! cargo build --release --example resize
//! leaks --atExit -- target/release/examples/resize 30
//! ```

use metalfx_wgpu::{DenoiserDescriptor, DenoiserParams, DenoiserTextures, MetalFxDenoiser};

fn main() {
    let cycles: u32 = std::env::args().nth(1).map_or(10, |s| s.parse().unwrap());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e:#?}")));

    let sizes = [
        ([1280, 720], [1280, 720]),
        ([1920, 1080], [1920, 1080]),
        ([960, 540], [1920, 1080]),
    ];
    for cycle in 0..cycles {
        for &(input, output) in &sizes {
            if std::env::var_os("POOL").is_some() {
                objc2::rc::autoreleasepool(|_| run(&device, &queue, input, output));
            } else {
                run(&device, &queue, input, output);
            }
        }
        println!("cycle {cycle}: rss {} MB", rss_mb());
    }
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, input: [u32; 2], output: [u32; 2]) {
    {
        let desc = DenoiserDescriptor::solari(input, output);
        let mut denoiser = MetalFxDenoiser::new(device, desc).unwrap();
        let usages = denoiser.required_usages();
        let make = |size: [u32; 2], format, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let textures = [
            make(input, desc.color_format, usages.color),
            make(input, desc.depth_format, usages.depth),
            make(input, desc.motion_format, usages.motion),
            make(input, desc.diffuse_albedo_format, usages.diffuse_albedo),
            make(input, desc.specular_albedo_format, usages.specular_albedo),
            make(input, desc.normal_format, usages.normal),
            make(input, desc.roughness_format, usages.roughness),
            make(output, desc.output_format, usages.output),
        ];
        // Initialize everything through wgpu (see the crate docs).
        let mut encoder = device.create_command_encoder(&Default::default());
        for t in &textures {
            let view = t.create_view(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        queue.submit([encoder.finish()]);

        let [
            color,
            depth,
            motion,
            diffuse_albedo,
            specular_albedo,
            normal,
            roughness,
            out,
        ] = &textures;
        let t = DenoiserTextures {
            color,
            depth,
            motion,
            diffuse_albedo,
            specular_albedo,
            normal,
            roughness,
            output: out,
        };
        for frame in 0..3 {
            let params = DenoiserParams {
                reset: frame == 0,
                ..Default::default()
            };
            queue.submit([denoiser.encode(device, t, &params).unwrap()]);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
}

fn rss_mb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        / 1024
}
