use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread, warp};

type AppResult<T> = Result<T, Box<dyn Error>>;

#[cuda_module]
mod kernels {
    use super::*;

    const GROUP_SIZE: usize = 16;
    const FP4_MAX: f32 = 6.0;
    const FP8_MAX_FOUR_SIX: f32 = 256.0;

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        scale_override: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let lane_in_group = lane & 0x0f;
        let group_mask = if lane < GROUP_SIZE {
            0x0000_ffff
        } else {
            0xffff_0000
        };
        let group_leader = (lane & !0x0f) as u32;
        let groups_per_block = thread::blockDim_x() as usize / GROUP_SIZE;
        let group = thread::blockIdx_x() as usize * groups_per_block
            + thread::threadIdx_x() as usize / GROUP_SIZE;

        if group < out_scales.len() {
            let base = group * GROUP_SIZE;
            let tensor_amax = amax[0];
            let global_scale = if tensor_amax == 0.0 {
                1.0
            } else {
                tensor_amax * scale_override / (FP8_MAX_FOUR_SIX * FP4_MAX)
            };

            unsafe {
                if group == 0 && lane_in_group == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }

                let value = x[base + lane_in_group];
                let group_amax = half_warp_max(abs_f32(value), group_mask);
                let mut scale_bits_six = 0u16;
                let mut scale_bits_four = 0u16;
                let mut scale_six = 0.0;
                let mut scale_four = 0.0;

                if lane_in_group == 0 {
                    scale_bits_six =
                        local_scale_bits(group_amax, global_scale, scale_override, 6.0);
                    scale_bits_four =
                        local_scale_bits(group_amax, global_scale, scale_override, 4.0);
                    scale_six = local_scale_value(scale_bits_six);
                    scale_four = local_scale_value(scale_bits_four);
                }

                scale_bits_six =
                    warp::shuffle_sync(group_mask, scale_bits_six as u32, group_leader) as u16;
                scale_bits_four =
                    warp::shuffle_sync(group_mask, scale_bits_four as u32, group_leader) as u16;
                scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
                scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);

                let err_six =
                    half_warp_sum(candidate_error(value, scale_six, global_scale), group_mask);
                let err_four =
                    half_warp_sum(candidate_error(value, scale_four, global_scale), group_mask);
                let grid_max = if err_six <= err_four { 6.0 } else { 4.0 };
                let scale_bits = if grid_max == 6.0 {
                    scale_bits_six
                } else {
                    scale_bits_four
                };
                let scale = if grid_max == 6.0 {
                    scale_six
                } else {
                    scale_four
                };
                let scale_for_payload = if scale == 0.0 { 1.0 } else { scale };
                let inv_scale = 1.0 / (scale_for_payload * global_scale);

                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits as u8;
                }

                if lane_in_group < GROUP_SIZE / 2 {
                    let pair = lane_in_group * 2;
                    let hi = x[base + pair] * inv_scale;
                    let lo = x[base + pair + 1] * inv_scale;
                    *out_fp4.get_unchecked_mut(base / 2 + lane_in_group) =
                        cvt_rn_satfinite_e2m1x2_f32(hi, lo);
                }
            }
        }
    }

    #[inline(always)]
    fn half_warp_max(mut value: f32, mask: u32) -> f32 {
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 8));
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 4));
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 2));
        max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 1))
    }

    #[inline(always)]
    fn half_warp_sum(mut value: f32, mask: u32) -> f32 {
        value += warp::shuffle_xor_f32_sync(mask, value, 8);
        value += warp::shuffle_xor_f32_sync(mask, value, 4);
        value += warp::shuffle_xor_f32_sync(mask, value, 2);
        value + warp::shuffle_xor_f32_sync(mask, value, 1)
    }

    #[inline(always)]
    fn candidate_error(value: f32, scale: f32, global_scale: f32) -> f32 {
        let scale_for_payload = if scale == 0.0 { 1.0 } else { scale };
        let inv_scale = 1.0 / (scale_for_payload * global_scale);
        let dequant_scale = scale * global_scale;
        let packed = cvt_rn_satfinite_e2m1x2_f32(0.0, value * inv_scale);
        let dequant = e2m1_value(packed & 0x0f) * dequant_scale;
        let diff = value - dequant;
        diff * diff
    }

    #[inline(always)]
    fn local_scale_value(bits: u16) -> f32 {
        let value: f32;
        unsafe {
            ptx_asm!(
                "{ .reg .b32 h2; .reg .b16 lo; cvt.rn.f16x2.e4m3x2 h2, %1; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
                out("=f") value,
                in("h") bits,
                options(register_only),
            );
        }
        value
    }

    #[inline(always)]
    fn local_scale_bits(
        group_amax: f32,
        global_scale: f32,
        scale_override: f32,
        grid_max: f32,
    ) -> u16 {
        let value = group_amax * scale_override / (grid_max * global_scale);
        let packed: u16;

        unsafe {
            ptx_asm!(
                "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;",
                out("=h") packed,
                in("f") 0.0f32,
                in("f") value,
                options(register_only),
            );
        }
        packed
    }

    #[inline(always)]
    fn cvt_rn_satfinite_e2m1x2_f32(hi: f32, lo: f32) -> u8 {
        let packed: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u16.u8 %0, tmp; }",
                out("=h") packed,
                in("f") hi,
                in("f") lo,
                options(register_only),
            );
        }
        packed as u8
    }

    #[inline(always)]
    fn e2m1_value(bits: u8) -> f32 {
        let value: f32;
        let packed = bits as u16;

        unsafe {
            ptx_asm!(
                "{ .reg .b8 e2; .reg .b32 h2; .reg .b16 lo; cvt.u8.u16 e2, %1; cvt.rn.f16x2.e2m1x2 h2, e2; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
                out("=f") value,
                in("h") packed,
                options(register_only),
            );
        }
        value
    }

    #[inline(always)]
    fn abs_f32(x: f32) -> f32 {
        let y: f32;
        unsafe {
            ptx_asm!(
                "abs.f32 %0, %1;",
                out("=f") y,
                in("f") x,
                options(register_only),
            );
        }
        y
    }

    #[inline(always)]
    fn max_f32(a: f32, b: f32) -> f32 {
        let y: f32;
        unsafe {
            ptx_asm!(
                "max.f32 %0, %1, %2;",
                out("=f") y,
                in("f") a,
                in("f") b,
                options(register_only),
            );
        }
        y
    }
}

pub fn run_default() -> AppResult<()> {
    let x = [
        -3.25f32, -2.0, -1.25, -0.5, -0.125, 0.0, 0.25, 0.75, 1.0, 1.5, 2.25, 3.0, 4.0, 5.0, 6.5,
        8.0,
    ];
    let amax = [x.iter().fold(0.0f32, |max, v| max.max(v.abs()))];

    let ctx = CudaContext::new(1)?;
    let stream = ctx.new_stream()?;
    let module = kernels::from_module(ctx.load_module_from_file(crate::CUDA_OXIDE_PTX_PATH)?)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let group_count = (x.len() / 16) as u32;
    let threads_per_block = 256u32;
    let groups_per_block = threads_per_block / 16;

    module.fp32_to_nvfp4_four_six_kernel(
        &stream,
        LaunchConfig {
            grid_dim: (group_count.div_ceil(groups_per_block), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        },
        &x_dev,
        &amax_dev,
        &mut fp4_dev,
        &mut scales_dev,
        &mut global_scale_dev,
        1.0f32,
    )?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scale = global_scale_dev.to_host_vec(&stream)?;
    println!(
        "nvfp4 fp4=[{}] scales=[{}] global_scale={:.8e}",
        hex_bytes(&fp4),
        hex_bytes(&scales),
        global_scale[0]
    );
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
