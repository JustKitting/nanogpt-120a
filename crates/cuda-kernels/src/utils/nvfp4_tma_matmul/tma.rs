use std::ffi::c_void;
use std::mem::MaybeUninit;

use cuda_core::{DeviceBuffer, DriverError};

use super::cute::{TmaSwizzle, U4SmemLayout};

pub struct TmaNvfp4DeviceScaleDescriptors {
    pub a: DeviceBuffer<[u64; 16]>,
    pub b: DeviceBuffer<[u64; 16]>,
    pub a_scales: DeviceBuffer<[u64; 16]>,
    pub b_scales: DeviceBuffer<[u64; 16]>,
}

pub fn encode_u4_tiled_layout<L: U4SmemLayout>(
    global_address: *mut c_void,
    global_width_u4: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_height: u32,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;

    encode_tiled(
        global_address,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        global_width_u4,
        global_height,
        row_stride_bytes,
        L::PACKS_PER_ROW * 8,
        tile_height,
        L::TMA_SWIZZLE,
    )
}

pub fn encode_u16_tiled(
    global_address: *mut c_void,
    global_width_u16: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_width_u16: u32,
    tile_height: u32,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16;

    encode_tiled(
        global_address,
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16,
        global_width_u16,
        global_height,
        row_stride_bytes,
        tile_width_u16,
        tile_height,
        TmaSwizzle::None,
    )
}

fn encode_tiled(
    global_address: *mut c_void,
    data_type: cuda_core::sys::CUtensorMapDataType_enum,
    global_width: u64,
    global_height: u64,
    row_stride_bytes: u64,
    tile_width: u32,
    tile_height: u32,
    swizzle: TmaSwizzle,
) -> Result<[u64; 16], DriverError> {
    use cuda_core::sys::{
        CUtensorMap, CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim = [global_width, global_height];
    let global_strides = [row_stride_bytes];
    let box_dim = [tile_width, tile_height];
    let element_strides = [1, 1];
    let cu_swizzle = match swizzle {
        TmaSwizzle::None => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
        TmaSwizzle::Swizzle64B => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        TmaSwizzle::Swizzle128B => CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
    };

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            data_type,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu_swizzle,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(DriverError(result));
    }

    let tensor_map = unsafe { tensor_map.assume_init() };
    Ok(unsafe { std::mem::transmute_copy::<CUtensorMap, [u64; 16]>(&tensor_map) })
}
