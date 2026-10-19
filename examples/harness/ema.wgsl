// Naive temporal accumulation baseline: reproject the history with the motion vectors (bilinear)
// and blend in the current frame. No neighborhood clamping, no disocclusion detection beyond
// "reprojected off screen".

struct Params {
    alpha: f32,
    reset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var history_sampler: sampler;
@group(0) @binding(5) var history_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(current);
    if any(id.xy >= size) {
        return;
    }
    let color = textureLoad(current, id.xy, 0).rgb;
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
    // Bevy motion vectors are current - previous, in UV units.
    let previous_uv = uv - textureLoad(motion, id.xy, 0).xy;
    var out = color;
    if params.reset == 0u && all(previous_uv >= vec2f(0.0)) && all(previous_uv <= vec2f(1.0)) {
        let past = textureSampleLevel(history, history_sampler, previous_uv, 0.0).rgb;
        out = mix(past, color, params.alpha);
    }
    textureStore(history_out, id.xy, vec4f(out, 1.0));
}
