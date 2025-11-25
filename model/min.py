import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# --- CẤU HÌNH (CONFIG) ---
DATA_DIR = "./data_clean" # Folder chứa data chuẩn
BATCH_SIZE = 32         # CPU yếu thì để 32 hoặc 16 thôi
N_EPOCHS = 10           # Chạy thử 5-10 epoch xem sao
CLIP = 1.0              # Cắt gradient để không bị lỗi
LR = 0.001              # Tốc độ học

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Đang chạy trên: {DEVICE}")

# --- 1. CHUẨN BỊ DỮ LIỆU ---
print(" Đang load dữ liệu...")
train_dataset = Multi30kDataset(DATA_DIR, mode='train')
val_dataset = Multi30kDataset(DATA_DIR, mode='val', src_vocab=train_dataset.src_vocab, trg_vocab=train_dataset.trg_vocab)
# test_dataset = Multi30kDataset(DATA_DIR, mode='test', src_vocab=train_dataset.src_vocab, trg_vocab=train_dataset.trg_vocab)

pad_idx = train_dataset.src_vocab.stoi["<pad>"]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collate(pad_idx), num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(pad_idx), num_workers=0)

print(f" Data ngon lành! Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# --- 2. KHỞI TẠO MODEL ---
INPUT_DIM = len(train_dataset.src_vocab)
OUTPUT_DIM = len(train_dataset.trg_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = MyTranslationModel(enc, dec, DEVICE).to(DEVICE)

# Khởi tạo trọng số (Weight Initialization) - Giúp model học nhanh hơn
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)
# Bỏ qua loss của token <pad>
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# --- 3. HÀM TRAIN & EVALUATE ---
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()

        # output: [trg len, batch size, output dim]
        output = model(src, trg)

        output_dim = output.shape[-1]

        # Bỏ token <sos> đầu tiên khi tính loss
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Clip gradient để tránh bùng nổ
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        # Log nhẹ cái để biết máy không bị treo (mỗi 50 batch báo 1 lần)
        if i % 50 == 0:
            print(f"   Batch {i}/{len(iterator)} | Loss: {loss.item():.3f}")

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0) # Tắt teacher forcing khi val/test
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# --- 4. HÀM DỊCH THỬ (LẤY ĐIỂM) ---
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    # 1. Tokenize & Numericalize
    if isinstance(sentence, str):
        import spacy
        spacy_en = spacy.load("en_core_web_sm")
        tokens = [token.text.lower() for token in spacy_en(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ["<sos>"] + tokens + ["<eos>"]
    src_indexes = [src_vocab.stoi.get(token, src_vocab.stoi["<unk>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device) # [len, 1]

    with torch.no_grad():
        # Encoder
        hidden, cell = model.encoder(src_tensor)

    # Decoder
    trg_indexes = [trg_vocab.stoi["<sos>"]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device) # [1]

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.stoi["<eos>"]:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:-1] # Bỏ sos, eos

# --- 5. VÒNG LẶP CHÍNH (MAIN LOOP) ---
if __name__ == "__main__":
    import os

    # 1. CẤU HÌNH
    DATA_DIR = "data_clean"

    # --- KIỂM TRA AN TOÀN ---
    # Nếu không thấy data thì báo lỗi dừng lại, CHỨ KHÔNG TỰ TẠO FILE MỚI NỮA
    if not os.path.exists(os.path.join(DATA_DIR, 'train.en')):
        print(" LỖI TO: Không tìm thấy file dữ liệu!")
        print(" Chạy lại Bước 1 (Cell Cứu Hộ) để tải data về đã!")
        exit() # Dừng chương trình ngay lập tức

    # 2. Load Dataset (Chỉ đọc)
    print(f" Đang đọc dữ liệu từ {DATA_DIR}...")
    train_dataset = Multi30kDataset(DATA_DIR, mode='train')
    val_dataset = Multi30kDataset(DATA_DIR, mode='val', src_vocab=train_dataset.src_vocab, trg_vocab=train_dataset.trg_vocab)

    # Kiểm tra số lượng (Phải ~29000 mới đúng)
    print(f" Số lượng câu Train: {len(train_dataset)}")
    print(f" Số lượng câu Val:   {len(val_dataset)}")
    print(f" Vocab Size: Anh={len(train_dataset.src_vocab)}, Pháp={len(train_dataset.trg_vocab)}")

    # 3. Setup DataLoader
    pad_idx = train_dataset.src_vocab.stoi["<pad>"]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collate(pad_idx), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(pad_idx), num_workers=0)

    # 4. Khởi tạo Model
    INPUT_DIM = len(train_dataset.src_vocab)
    OUTPUT_DIM = len(train_dataset.trg_vocab)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = MyTranslationModel(enc, dec, DEVICE).to(DEVICE)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 5. Bắt đầu Train
    best_valid_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n Bắt đầu Train {N_EPOCHS} epochs (Dữ liệu thật)...")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"     Đã lưu model tốt nhất (Val Loss: {valid_loss:.3f})")

        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    # 6. Vẽ biểu đồ & Test
    print("\n Đang vẽ biểu đồ Loss...")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('loss_chart.png')
    plt.show()

    model.load_state_dict(torch.load('best_model.pth'))
    print("\n Dịch thử 1 câu:")
    sentence = "A man is walking a dog."
    translation = translate_sentence(sentence, train_dataset.src_vocab, train_dataset.trg_vocab, model, DEVICE)
    print(f"Src: {sentence}")
    print(f"Trg: {' '.join(translation)}")