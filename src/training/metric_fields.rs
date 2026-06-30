macro_rules! metric_fields {
    (
        $field_ty:ident,
        $field_const:ident,
        $spec_ty:ident,
        $prefix:literal
        {
            $($variant:ident => ($name:literal, $unit:expr, $higher_is_better:expr),)+
        }
    ) => {
        #[derive(Clone, Copy)]
        enum $field_ty {
            $($variant,)+
        }

        const $field_const: &[$field_ty] = &[
            $($field_ty::$variant,)+
        ];

        impl $field_ty {
            const fn spec(self) -> $spec_ty {
                match self {
                    $(
                        Self::$variant => $spec_ty {
                            name: concat!($prefix, $name),
                            unit: $unit,
                            higher_is_better: $higher_is_better,
                            field: Self::$variant,
                        },
                    )+
                }
            }
        }
    };
}
