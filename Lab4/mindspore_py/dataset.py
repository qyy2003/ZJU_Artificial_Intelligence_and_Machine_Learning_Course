import os
import mindspore.dataset as ds

def create_dataset(mindrecord_path, cfg):
    """Data operations."""
    ds.config.set_seed(1)
    file_name = os.path.join(mindrecord_path, 'style.mindrecord0')
    data_set = ds.MindDataset(file_name, columns_list=["feature", "label"], num_parallel_workers=4)

    train_ds, eval_ds = data_set.split(cfg.data_split, randomize=True)
    train_ds = train_ds.shuffle(buffer_size=train_ds.get_dataset_size()//5)
    train_ds = train_ds.batch(batch_size=cfg.batch_size, drop_remainder=True)
    eval_ds = eval_ds.batch(batch_size=cfg.batch_size, drop_remainder=True)
    return train_ds, eval_ds
