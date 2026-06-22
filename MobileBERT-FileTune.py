import pandas as pd
import numpy as np
import matplotlib
import platform
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from torch.optim import AdamW
# 한국어 처리를 위해 다국어 BERT 토크나이저 및 모델 사용
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

# 한글 깨짐 방지를 위한 한글 폰트 설정 (Windows의 맑은 고딕 기준)
if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # 맥(Mac) 환경인 경우
    matplotlib.rc("font", family="AppleGothic")
matplotlib.rc("axes", unicode_minus=False)  # 마이너스 기호 깨짐 방지

# ────────────────────────────────────────────────
# 0. Config
# ────────────────────────────────────────────────
MAX_SAMPLES = 3000
BATCH_SIZE = 8
EPOCH = 4
MAX_LENGTH = 128  # ratings.txt는 단문 리뷰가 많으므로 효율성을 위해 128로 조절 (필요시 512 변경 가능)
LR = 2e-5
RANDOM_STATE = 2026
OUTPUT_DIR = "ratings_bert"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# 1. Plot helper functions
# ────────────────────────────────────────────────
def plot_label_distribution(labels, save_path):
    """라벨 분포 파이 차트 + 바 차트"""
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ["부정적 반응 (0)", "긍정적 반응 (1)"]
    colors = ["#FF6B6B", "#4ECDC4"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("네이버 영화 리뷰 - 호불호(라벨) 분포", fontsize=15, fontweight="bold")

    # 파이 차트
    axes[0].pie(counts, labels=label_names, autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0.05, 0.05))
    axes[0].set_title("라벨 비율")

    # 바 차트
    bars = axes[1].bar(label_names, counts, color=colors, edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, f"{cnt}개",
                     ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("데이터 개수")
    axes[1].set_title("라벨별 샘플 개수")
    axes[1].set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[저장 완료] {save_path}")
    plt.show()
    plt.close()


def plot_training_results(epoch_results, save_path):
    """[수정본] 데이터 개수가 맞지 않아도 안전하게 그리는 에포크별 Loss 및 정확도 추이 그래프"""
    if not epoch_results:
        print("[시각화 실패] 누적된 에포크 결과 데이터가 없습니다.")
        return

    # 💡 핵심 수정: X축 범위를 고정된 EPOCH이 아니라, 실제 저장된 데이터 개수만큼만 동적으로 설정합니다.
    losses = [x[0] for x in epoch_results]
    train_accs = [x[1] for x in epoch_results]
    valid_accs = [x[2] for x in epoch_results]
    epochs = range(1, len(losses) + 1)  # (4,)와 (3,)으로 충돌나던 문제를 원천 차단

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BERT 모델 학습 프로세스 결과 성능 지표", fontsize=15, fontweight="bold")

    # 1. Loss 추이 그래프
    axes[0].plot(epochs, losses, marker='o', color='#E63946', linewidth=2, label='훈련 손실값 (Loss)')
    axes[0].set_title('에포크별 손실값 추이 (Loss Train Trend)')
    axes[0].set_xlabel('에포크 (Epoch)')
    axes[0].set_ylabel('손실값 (Loss)')
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # 2. Accuracy 추이 그래프 (학습 정확도 vs 검증 정확도)
    axes[1].plot(epochs, train_accs, marker='o', color='#457B9D', linewidth=2, label='학습 정확도 (Train Acc)')
    axes[1].plot(epochs, valid_accs, marker='s', color='#1D3557', linewidth=2, linestyle='--',
                 label='검증 정확도 (Valid Acc)')
    axes[1].set_title('에포크별 정확도 비교 추이 (Accuracy Trend)')
    axes[1].set_xlabel('에포크 (Epoch)')
    axes[1].set_ylabel('정확도 (Accuracy)')
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[저장 완료] {save_path}")

    # 💡 만약 GUI 창 대기 때문에 멈추는 걸 방지하려면 아래 plt.show()를 주석 처리(#)해도 좋습니다.
    plt.show()
    plt.close()


# ────────────────────────────────────────────────
# 2. Main
# ────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    logging.set_verbosity_error()

    # -- 데이터 불러오기 --
    path = "ratings.txt"
    df = pd.read_csv(path, sep="\t")

    df = df.dropna(subset=["document", "label"]).reset_index(drop=True)
    print(f"\n원본 데이터셋 크기: {len(df):,}")

    # -- 원본 비율 그대로 무작위 샘플링 --
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

    n_pos = len(df[df["label"] == 1])
    n_neg = len(df[df["label"] == 0])
    print(f"샘플링 후 크기: {len(df):,} (긍정 {n_pos}개, 부정 {n_neg}개)")

    text = list(df["document"].values)
    labels = df["label"].values

    print("\n=== 데이터 미리보기 ===")
    print("리뷰 텍스트 :", text[:3])
    print("감성 라벨   :", labels[:5])

    # -- 시각화 실행 (라벨 분포 그래프 출력) --
    print("\n[시각화] 라벨 분포 출력 중...")
    plot_label_distribution(labels, f"{PLOT_DIR}/01_label_distribution.png")

    # -- 토크나이저 --
    print("\n토큰화 작업 진행 중...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding="max_length"
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("\n=== 토큰화 샘플 데이터 ===")
    for j in range(2):
        print(f"\n샘플 {j + 1}")
        print("토큰 ID (앞 10개)    :", input_ids[j][:10], "...")
        print("어텐션 마스크 (앞 10개):", attention_mask[j][:10], "...")

    # -- 데이터셋 분할 (학습용 / 검증용) --
    tx, vx, ty, vy = train_test_split(input_ids, labels, test_size=0.2, random_state=RANDOM_STATE)
    tm, vm, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=RANDOM_STATE)

    # -- 데이터로더 생성 --
    def make_loader(ids, masks, lbls, sampler_cls):
        ds = TensorDataset(
            torch.tensor(ids),
            torch.tensor(masks),
            torch.tensor(lbls)
        )
        return DataLoader(ds, sampler=sampler_cls(ds), batch_size=BATCH_SIZE)

    train_dataloader = make_loader(tx, tm, ty, RandomSampler)
    valid_dataloader = make_loader(vx, vm, vy, SequentialSampler)

    print(f"\n학습 배치 수: {len(train_dataloader)}, 검증 배치 수: {len(valid_dataloader)}")

    # -- 모델 초기화 및 학습 설정 --
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_dataloader) * EPOCH
    num_warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # -- 학습 루프 시작 --
    epoch_results = []

    for e in range(EPOCH):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"에포크 진행 중 {e + 1}/{EPOCH}", leave=False)

        for batch in pbar:
            batch = tuple(t.to(device) for t in batch)
            b_ids, b_masks, b_labels = batch
            model.zero_grad()
            outputs = model(b_ids, attention_mask=b_masks, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # 학습 데이터 정확도 측정
        model.eval()
        train_preds, train_true = [], []
        for batch in tqdm(train_dataloader, desc=f"  학습 정확도 측정 {e + 1}", leave=False):
            batch = tuple(t.to(device) for t in batch)
            b_ids, b_masks, b_labels = batch
            with torch.no_grad():
                logits = model(b_ids, attention_mask=b_masks).logits
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_true.extend(b_labels.cpu().numpy())
        train_acc = np.mean(np.array(train_preds) == np.array(train_true))

        # 검증 데이터 정확도 측정
        valid_preds, valid_true = [], []
        for batch in tqdm(valid_dataloader, desc=f"  검증 정확도 측정 {e + 1}", leave=False):
            batch = tuple(t.to(device) for t in batch)
            b_ids, b_masks, b_labels = batch
            with torch.no_grad():
                logits = model(b_ids, attention_mask=b_masks).logits
            valid_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            valid_true.extend(b_labels.cpu().numpy())
        valid_acc = np.mean(np.array(valid_preds) == np.array(valid_true))

        epoch_results.append((avg_train_loss, train_acc, valid_acc))
        print(f"\n에포크 {e + 1}: 손실값(Loss)={avg_train_loss:.4f}  "
              f"학습 정확도={train_acc * 100:.2f}%  검증 정확도={valid_acc * 100:.2f}%")

    # -- 결과 종합 출력 --
    print("\n=== 최종 학습 결과 요약 ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, 1):
        print(f"에포크 {idx}: 손실={loss:.4f}  학습 정확도={tacc:.4f}  검증 정확도={vacc:.4f}")

    # -- [신규 반영] 에포크 통합 결과 시각화 차트 생성 및 저장 --
    print("\n[시각화] 에포크별 모델 성능 지표 곡선 출력 중...")
    plot_training_results(epoch_results, f"{PLOT_DIR}/02_training_metrics.png")

    # -- 모델 저장 --
    print("\n=== 학습 완료된 모델 저장 ===")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"모델이 성공적으로 {OUTPUT_DIR}/ 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()