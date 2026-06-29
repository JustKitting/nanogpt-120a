pub(crate) const THREADS_PER_BLOCK: u32 = 256;
pub(crate) const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / 32;
pub(crate) const GROUP_SIZE_U32: u32 = 16;
pub(crate) const GROUPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / GROUP_SIZE_U32;
