macro_rules! metric_fields {
    (
        $field_ty:ident,
        $field_const:ident,
        $specs_fn:ident,
        $spec_ty:ident,
        $prefix:literal
        {
            $($variant:ident => ($name:literal, $unit:expr, $higher_is_better:expr),)+
        }
    ) => {
        #[derive(Clone, Copy)]
        pub(super) struct $spec_ty {
            name: &'static str,
            unit: Option<&'static str>,
            higher_is_better: bool,
            field: $field_ty,
        }

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

        pub(super) fn $specs_fn() -> impl Iterator<Item = $spec_ty> {
            $field_const.iter().copied().map($field_ty::spec)
        }
    };
}

macro_rules! impl_numeric_metric_spec {
    ($spec_ty:ty, $input_ty:ty, $value:expr $(,)?) => {
        impl $crate::training::numeric_metric::NumericMetricSpec for $spec_ty {
            type Input = $input_ty;

            fn name(self) -> &'static str {
                self.name
            }

            fn unit(self) -> Option<&'static str> {
                self.unit
            }

            fn higher_is_better(self) -> bool {
                self.higher_is_better
            }

            fn value(self, item: &Self::Input) -> f64 {
                $value(self.field, item)
            }
        }
    };
}
