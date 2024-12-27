import os
import re

# this function finds the latest checkpoint from `checkpoint_path`.
# checkpoint files are in the format of `epoch={epoch}-step={step}.ckpt`.
def find_latest_checkpoint(checkpoint_path):
    def extract_epoch_and_step(filename):
        m = re.match(r"epoch=(\d+)-step=(\d+).ckpt", filename)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2))

    def compare_epoch_and_step(a, b):
        assert a is not None
        if b is None:
            return True
        return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1])

    latest_epoch_and_step = None
    latest_checkpoint = None
    for filename in os.listdir(checkpoint_path):
        epoch_and_step = extract_epoch_and_step(filename)
        if compare_epoch_and_step(epoch_and_step, latest_epoch_and_step):
            latest_epoch_and_step = epoch_and_step
            latest_checkpoint = filename
    return latest_checkpoint