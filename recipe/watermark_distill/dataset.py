"""
Custom dataset and collator for watermark distillation.

WatermarkSFTDataset extends SFTDataset to also return per-sample `wm_seed`
and `wm_fraction` read from the parquet data columns "seed" and "fraction".

WatermarkCollator extends SFTTensorCollator to properly handle scalar tensor
fields (stacked as regular tensors) alongside variable-length sequence tensors
(nested tensors).
"""

import torch

from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.sft_dataset import SFTDataset


class WatermarkSFTDataset(SFTDataset):
    """SFTDataset that also returns per-sample watermark seed and fraction."""

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        self.wm_seeds = self.dataframe["seed"].tolist()
        self.wm_fractions = self.dataframe["fraction"].tolist()

    def __getitem__(self, item):
        result = super().__getitem__(item)
        result["wm_seed"] = torch.tensor(int(self.wm_seeds[item]), dtype=torch.long)
        result["wm_fraction"] = torch.tensor(float(self.wm_fractions[item]), dtype=torch.float32)
        return result


class WatermarkCollator(SFTTensorCollator):
    """Collator that nests variable-length tensors but stacks scalar tensors."""

    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:
        final_batch = {}
        tensor_keys = [key for key in batch[0].keys() if isinstance(batch[0][key], torch.Tensor)]
        for key in tensor_keys:
            tensors = [item[key] for item in batch]
            if tensors[0].dim() == 0:
                final_batch[key] = torch.stack(tensors)
            else:
                final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        return final_batch
