import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from step2_dataset import get_dataloaders

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class VanillaRNN_BC(nn.Module):
    """
    SO-101 6-DoF 시계열 모방학습(BC) 바닐라 RNN 신경망
    - Projector: 6차원 연속 관절각 -> embed_dim(64) 차원 잠재 공간 선형 투영
    - RNN Layer: (embed_dim -> hidden_dim=128) 시계열 순환 누적
    - Context Vector: 마지막 타임스텝 은닉 상태 (out[:, -1, :]) 추출
    - Action Head: hidden_dim(128) -> 6차원 목표 관절각 예측
    """
    def __init__(self, input_dim=6, embed_dim=64, hidden_dim=128, output_dim=6, num_layers=1):
        super().__init__()
        
        # 1. 선형 투영 (Linear Projection): [Batch, History, 6] -> [Batch, History, 64]
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )
        
        # 2. RNN 순환 계층: [Batch, History, 64] -> [Batch, History, 128]
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3. 액션 헤드 (Linear Head): [Batch, 128] -> [Batch, 6]
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x Shape: [Batch, History, 6]
        
        # Step A. Projection
        proj = self.projector(x)       # Shape: [Batch, History, 64]
        
        # Step B. RNN Forward
        out, h_n = self.rnn(proj)      # out Shape: [Batch, History, 128], h_n: [1, Batch, 128]
        
        # Step C. Extract Context Vector (마지막 시점 은닉 상태)
        context = out[:, -1, :]        # Shape: [Batch, 128]
        
        # Step D. Action Prediction
        action_pred = self.head(context) # Shape: [Batch, 6]
        
        return action_pred

def train_model(
    data_path="demonstrations.npz",
    history_len=10,
    batch_size=64,
    epochs=35,
    lr=1e-3,
    save_path="best_rnn_bc_model.pth"
):
    print("=" * 60)
    print("🧠 [Step 3] 바닐라 RNN 시계열 BC 모델 학습 시작")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - 학습 디바이스: {device}")
    
    train_loader, test_loader = get_dataloaders(
        data_path=data_path,
        history_len=history_len,
        batch_size=batch_size
    )
    
    model = VanillaRNN_BC(
        input_dim=6,
        embed_dim=64,
        hidden_dim=128,
        output_dim=6
    ).to(device)
    
    # 텐서 차원 흐름 검증용 더미 입력
    dummy_x = torch.randn(2, history_len, 6).to(device)
    dummy_out = model(dummy_x)
    print(f"  - 모델 구조 확인 (더미 입력 {dummy_x.shape} -> 출력 {dummy_out.shape})")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_test_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_loader.dataset)
        
        # 2. Evaluation Phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                test_loss += loss.item() * len(batch_x)
                
        test_loss /= len(test_loader.dataset)
        
        # 모델 가중치 저장
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_loss": best_test_loss,
                "history_len": history_len,
                "input_dim": 6,
                "embed_dim": 64,
                "hidden_dim": 128,
                "output_dim": 6
            }, save_path)
            mark = "⭐️ [BEST]"
        else:
            mark = ""
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} {mark}")
            
    elapsed = time.time() - start_time
    print(f"\n✅ 모델 학습 완료! (소요 시간: {elapsed:.2f}초)")
    print(f"  - 최고 성능 Test MSE Loss : {best_test_loss:.6f}")
    print(f"  - 최적 가중치 저장 위치   : {os.path.abspath(save_path)}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
