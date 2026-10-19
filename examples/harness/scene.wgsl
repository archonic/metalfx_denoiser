// Small analytic scene for the denoiser harness: a ground plane, back wall, two spheres, a box and
// a spherical area light, under a dim sky.
//
// `noisy` renders one jittered path per pixel (one bounce, next-event estimation), plus guide
// buffers with Solari's formats and Bevy's conventions:
//   depth:  reverse-Z NDC depth of the jittered projection, 0 for sky
//   motion: (current_ndc - previous_ndc) * (0.5, -0.5) with unjittered matrices, as Bevy's prepass
// `reference` renders many box-filtered, unjittered paths per pixel as ground truth.

struct Uniforms {
    // Inverse of the projection used to generate primary rays (jittered for `noisy`).
    world_from_clip: mat4x4f,
    // Jittered, for depth.
    clip_from_world: mat4x4f,
    unjittered_clip_from_world: mat4x4f,
    previous_unjittered_clip_from_world: mat4x4f,
    view_from_world: mat4x4f,
    camera_position: vec4f,
    size: vec2u,
    frame: u32,
    samples: u32,
    normals_in_view_space: u32,
    light_offset: f32,
    _pad: vec2u,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depth_out: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var motion_out: texture_storage_2d<rg16float, write>;
@group(0) @binding(4) var diffuse_albedo_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var specular_albedo_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var normal_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var roughness_out: texture_storage_2d<r16float, write>;
@group(0) @binding(8) var reference_out: texture_storage_2d<rgba32float, write>;

const PI: f32 = 3.14159265;
const SKY: vec3f = vec3f(0.10, 0.14, 0.22);
const LIGHT_RADIUS: f32 = 0.5;
const LIGHT_RADIANCE: vec3f = vec3f(30.0, 27.0, 22.0);
const MISS: f32 = 1e30;

fn light_center() -> vec3f {
    return vec3f(1.5 + u.light_offset, 4.0, 2.0);
}

// ---------------------------------------------------------------------------------------------
// RNG (PCG)

var<private> rng_state: u32;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_state = pcg(rng_state);
    return f32(rng_state) / 4294967296.0;
}

// ---------------------------------------------------------------------------------------------
// Scene

struct Hit {
    t: f32,
    normal: vec3f,
    albedo: vec3f,
    roughness: f32,
    emissive: bool,
}

fn no_hit() -> Hit {
    return Hit(MISS, vec3f(0.0), vec3f(0.0), 1.0, false);
}

fn sphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> f32 {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;
    if h < 0.0 {
        return MISS;
    }
    let s = sqrt(h);
    let t0 = -b - s;
    if t0 > 1e-4 {
        return t0;
    }
    let t1 = -b + s;
    if t1 > 1e-4 {
        return t1;
    }
    return MISS;
}

fn aabb(ro: vec3f, rd: vec3f, lo: vec3f, hi: vec3f) -> vec4f {
    let inv = 1.0 / rd;
    let a = (lo - ro) * inv;
    let b = (hi - ro) * inv;
    let tmin = min(a, b);
    let tmax = max(a, b);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    if t_near > t_far || t_far < 1e-4 {
        return vec4f(MISS);
    }
    let t = select(t_far, t_near, t_near > 1e-4);
    let p = ro + rd * t;
    let center = (lo + hi) * 0.5;
    let d = (p - center) / ((hi - lo) * 0.5);
    let ad = abs(d);
    var n = vec3f(0.0, 0.0, sign(d.z));
    if ad.x > ad.y && ad.x > ad.z {
        n = vec3f(sign(d.x), 0.0, 0.0);
    } else if ad.y > ad.z {
        n = vec3f(0.0, sign(d.y), 0.0);
    }
    return vec4f(n, t);
}

fn trace(ro: vec3f, rd: vec3f) -> Hit {
    var hit = no_hit();

    // Ground plane y = 0, checkered.
    if rd.y < 0.0 {
        let t = -ro.y / rd.y;
        if t > 1e-4 && t < hit.t {
            let p = ro + rd * t;
            let checker = (i32(floor(p.x)) + i32(floor(p.z))) & 1;
            hit = Hit(t, vec3f(0.0, 1.0, 0.0), select(vec3f(0.75), vec3f(0.25), checker == 1), 0.8, false);
        }
    }
    // Back wall z = -4.
    if rd.z < 0.0 {
        let t = (-4.0 - ro.z) / rd.z;
        if t > 1e-4 && t < hit.t {
            hit = Hit(t, vec3f(0.0, 0.0, 1.0), vec3f(0.6, 0.55, 0.5), 0.9, false);
        }
    }

    let t_a = sphere(ro, rd, vec3f(0.0, 1.0, 0.0), 1.0);
    if t_a < hit.t {
        hit = Hit(t_a, normalize(ro + rd * t_a - vec3f(0.0, 1.0, 0.0)), vec3f(0.8, 0.2, 0.15), 0.5, false);
    }
    let t_b = sphere(ro, rd, vec3f(2.2, 0.6, -0.5), 0.6);
    if t_b < hit.t {
        hit = Hit(t_b, normalize(ro + rd * t_b - vec3f(2.2, 0.6, -0.5)), vec3f(0.2, 0.7, 0.3), 0.3, false);
    }
    let cube = aabb(ro, rd, vec3f(-2.5, 0.0, -1.0), vec3f(-1.3, 1.5, 0.5));
    if cube.w < hit.t {
        hit = Hit(cube.w, cube.xyz, vec3f(0.7, 0.7, 0.8), 0.6, false);
    }
    let t_l = sphere(ro, rd, light_center(), LIGHT_RADIUS);
    if t_l < hit.t {
        hit = Hit(t_l, normalize(ro + rd * t_l - light_center()), vec3f(1.0), 1.0, true);
    }
    return hit;
}

fn occluded(ro: vec3f, rd: vec3f, max_t: f32) -> bool {
    let hit = trace(ro, rd);
    return hit.t < max_t && !hit.emissive;
}

// ---------------------------------------------------------------------------------------------
// Shading

fn tangent_frame(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    return mat3x3f(t, bt, n);
}

// Direct light from the sphere light, sampled uniformly over its solid angle.
fn direct_light(p: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
    let to_light = light_center() - p;
    let dist2 = dot(to_light, to_light);
    let cos_max = sqrt(max(0.0, 1.0 - LIGHT_RADIUS * LIGHT_RADIUS / dist2));
    let cos_theta = 1.0 - rand() * (1.0 - cos_max);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * rand();
    let dir = tangent_frame(normalize(to_light)) * vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let n_dot_l = dot(n, dir);
    if n_dot_l <= 0.0 {
        return vec3f(0.0);
    }
    let origin = p + n * 1e-3;
    if occluded(origin, dir, sqrt(dist2)) {
        return vec3f(0.0);
    }
    let pdf = 1.0 / (2.0 * PI * (1.0 - cos_max));
    return LIGHT_RADIANCE * albedo / PI * n_dot_l / pdf;
}

fn cosine_sample(n: vec3f) -> vec3f {
    let r = sqrt(rand());
    let phi = 2.0 * PI * rand();
    return tangent_frame(n) * vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - r * r)));
}

// One path: primary hit, NEE, one cosine-sampled bounce with NEE at its hit.
fn radiance(ro: vec3f, rd: vec3f, primary: Hit) -> vec3f {
    if primary.t >= MISS {
        return SKY;
    }
    if primary.emissive {
        return LIGHT_RADIANCE;
    }
    let p = ro + rd * primary.t;
    var l = direct_light(p, primary.normal, primary.albedo);

    let bounce_dir = cosine_sample(primary.normal);
    let bounce_origin = p + primary.normal * 1e-3;
    let second = trace(bounce_origin, bounce_dir);
    if second.t >= MISS {
        l += primary.albedo * SKY;
    } else if !second.emissive {
        // Light hits on the bounce are skipped: NEE already counts them.
        let p2 = bounce_origin + bounce_dir * second.t;
        l += primary.albedo * direct_light(p2, second.normal, second.albedo);
    }
    return l;
}

fn primary_ray(pixel: vec2f) -> array<vec3f, 2> {
    let ndc = vec2f(pixel.x / f32(u.size.x) * 2.0 - 1.0, 1.0 - pixel.y / f32(u.size.y) * 2.0);
    // Reverse-Z: NDC z = 1 is the near plane.
    let near = u.world_from_clip * vec4f(ndc, 1.0, 1.0);
    let ro = u.camera_position.xyz;
    let rd = normalize(near.xyz / near.w - ro);
    return array<vec3f, 2>(ro, rd);
}

fn project(clip_from_world: mat4x4f, p: vec4f) -> vec3f {
    let clip = clip_from_world * p;
    return clip.xyz / clip.w;
}

@compute @workgroup_size(8, 8)
fn noisy(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= u.size) {
        return;
    }
    rng_state = pcg(id.x + pcg(id.y + pcg(u.frame)));

    // The jitter is in `world_from_clip`, so the pixel center is the jittered sample position.
    let ray = primary_ray(vec2f(id.xy) + 0.5);
    let ro = ray[0];
    let rd = ray[1];
    let hit = trace(ro, rd);
    let color = radiance(ro, rd, hit);

    var depth = 0.0;
    var motion = vec2f(0.0);
    var normal = vec3f(0.0);
    var albedo = vec3f(0.0);
    var roughness = 1.0;
    // Points at infinity (w = 0) for the sky, so camera rotation still gives motion.
    var world = vec4f(rd, 0.0);
    if hit.t < MISS {
        world = vec4f(ro + rd * hit.t, 1.0);
        depth = project(u.clip_from_world, world).z;
        normal = hit.normal;
        if u.normals_in_view_space != 0u {
            normal = normalize((u.view_from_world * vec4f(normal, 0.0)).xyz);
        }
        albedo = hit.albedo;
        roughness = hit.roughness;
    }
    let current = project(u.unjittered_clip_from_world, world).xy;
    let previous = project(u.previous_unjittered_clip_from_world, world).xy;
    motion = (current - previous) * vec2f(0.5, -0.5);

    textureStore(color_out, id.xy, vec4f(color, 1.0));
    textureStore(depth_out, id.xy, vec4f(depth));
    textureStore(motion_out, id.xy, vec4f(motion, 0.0, 0.0));
    textureStore(diffuse_albedo_out, id.xy, vec4f(albedo, 1.0));
    // Lambertian only: no specular reflectance.
    textureStore(specular_albedo_out, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(normal_out, id.xy, vec4f(normal, 0.0));
    textureStore(roughness_out, id.xy, vec4f(roughness));
}

@compute @workgroup_size(8, 8)
fn reference(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= u.size) {
        return;
    }
    rng_state = pcg(id.x + pcg(id.y + pcg(u.frame ^ 0x9e3779b9u)));
    var sum = vec3f(0.0);
    for (var i = 0u; i < u.samples; i++) {
        let ray = primary_ray(vec2f(id.xy) + vec2f(rand(), rand()));
        sum += radiance(ray[0], ray[1], trace(ray[0], ray[1]));
    }
    textureStore(reference_out, id.xy, vec4f(sum / f32(u.samples), 1.0));
}
