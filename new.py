import torch
import torchaudio
from src.liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO, device=device).eval()

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

# Generate text
for t in model.generate_sequential(**chat, max_new_tokens=512):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)