"""Standard demonstration SFT using the existing isolated FSDP trainer."""
import verl.trainer.sft_trainer as sft
from train_textcraft_sft import TextCraftSFTTrainer

if __name__ == '__main__':
    sft.SFTTrainer = TextCraftSFTTrainer
    sft.main()
