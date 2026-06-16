use cuda_device::ptx_asm;

#[inline(always)]
pub fn mma_m16n8k64_scale4x_ue4m3(
    a: [u32; 4],
    b: [u32; 2],
    acc: &mut [f32; 4],
    scale_a: u32,
    scale_b: u32,
) {
    unsafe {
        let d0 = acc.as_mut_ptr();
        let d1 = d0.add(1);
        let d2 = d0.add(2);
        let d3 = d0.add(3);

        ptx_asm!(
            "{
             .reg .f32 %%d0, %%d1, %%d2, %%d3;
             mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
             {%%d0, %%d1, %%d2, %%d3},
             {%4, %5, %6, %7},
             {%8, %9},
             {%10, %11, %12, %13},
             %14, {0, 0}, %15, {0, 0};
             st.f32 [%0], %%d0;
             st.f32 [%1], %%d1;
             st.f32 [%2], %%d2;
             st.f32 [%3], %%d3;
             }",
            in("l") d0,
            in("l") d1,
            in("l") d2,
            in("l") d3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b[0],
            in("r") b[1],
            in("f") acc[0],
            in("f") acc[1],
            in("f") acc[2],
            in("f") acc[3],
            in("r") scale_a,
            in("r") scale_b,
            clobber("memory"),
        );
    }
}
