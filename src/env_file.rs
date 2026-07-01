pub(crate) fn parsed<T: std::str::FromStr>(text: &str, name: &str) -> Option<T> {
    value(text, name)?.parse().ok()
}

pub(crate) fn value<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == name).then_some(value.trim())
    })
}
