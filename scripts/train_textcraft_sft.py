"""Standard full-parameter TextCraft SFT, preserving template and EOS IDs."""
import os
import verl.trainer.sft_trainer as sft
from dchl_dataset_support import FixedSupportMixin

class TextCraftSFTTrainer(FixedSupportMixin,sft.SFTTrainer):
    def fit(self):
        os.environ.pop('VES_HF_EXPORT_STEP',None)
        return super().fit()

if __name__=='__main__':
    sft.SFTTrainer=TextCraftSFTTrainer
    sft.main()
