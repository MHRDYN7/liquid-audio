import torch
import torchaudio
from src.liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device).eval()
# model = LFM2AudioModel.from_pretrained(HF_REPO, device=device).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform ASR.")
chat.end_turn()

chat.new_turn("user")
wav, sampling_rate = torchaudio.load("assets/asr.wav")
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# print(chat.model_inputs)
for k, v in chat.items():
    print(k, v)

# print(chat.proc.text.decode([chat.text[:,t] for t in range(len(chat.text[0]))]))
# print([chat.text[:,t] for t in range(len(chat.text[0]))])
# print(chat.proc.text.decode(torch.tensor([64015, 23])))
# print(chat.proc.text.decode(chat.text.squeeze(0)))
# print(processor.text_tokenizer)


# <|startoftext|><|im_start|>system
# Perform ASR.<|im_end|>
# <|im_start|>user
# <|im_end|>
# <|im_start|>assistant