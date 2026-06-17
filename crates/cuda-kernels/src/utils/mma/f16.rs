use cuda_device::ptx_asm;

#[inline(always)]
pub fn mma_m16n8k16_f16_f16_f32(a: [u32; 4], b: [u32; 2], acc: &mut [f32; 4]) {
    unsafe {
        let d0 = acc.as_mut_ptr();
        let d1 = d0.add(1);
        let d2 = d0.add(2);
        let d3 = d0.add(3);
        ptx_asm!(
            "{
             .reg .f32 %%d0, %%d1, %%d2, %%d3, %%c0, %%c1, %%c2, %%c3;
             mov.b32 %%c0, %10;
             mov.b32 %%c1, %11;
             mov.b32 %%c2, %12;
             mov.b32 %%c3, %13;
             mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
             {%%d0, %%d1, %%d2, %%d3},
             {%4, %5, %6, %7},
             {%8, %9},
             {%%c0, %%c1, %%c2, %%c3};
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
            clobber("memory"),
        );
    }
}
