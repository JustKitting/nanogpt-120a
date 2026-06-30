use burn::data::dataloader::Progress;

pub(super) const TRAIN_EPOCH: usize = 1;

pub(super) fn epoch_progress() -> Progress {
    Progress::new(TRAIN_EPOCH, TRAIN_EPOCH)
}
