import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class TimeSeriesBCDataset(Dataset):
    """
    SO-101 6-DoF 시계열 모방학습(BC) 슬라이딩 윈도우 Dataset
    - 입력 (X): 최근 H개 프레임의 관절 상태 [H, 6]
    - 출력 (Y): 다음 1스텝(t+1)의 목표 액션 [6]
    """
    def __init__(self, data_path="demonstrations.npz", history_len=10, is_train=True, train_ratio=0.8):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Demonstration data not found at: {data_path}")
            
        data = np.load(data_path)
        states = data["states"]    # [100, 100, 6]
        actions = data["actions"]  # [100, 100, 6]
        
        num_episodes, episode_len, num_dof = states.shape
        num_train = int(num_episodes * train_ratio)
        
        if is_train:
            states = states[:num_train]     # [80, 100, 6]
            actions = actions[:num_train]   # [80, 100, 6]
        else:
            states = states[num_train:]    # [20, 100, 6]
            actions = actions[num_train:]  # [20, 100, 6]
            
        self.samples_X = []
        self.samples_Y = []
        
        # ⚠️ 중요: 서로 다른 에피소드가 하나의 윈도우에 섞이지 않도록 에피소드별로 분리 슬라이딩
        for ep in range(len(states)):
            ep_states = states[ep]   # [100, 6]
            ep_actions = actions[ep] # [100, 6]
            
            for t in range(history_len - 1, episode_len - 1):
                # t-H+1 부터 t 까지 (총 H개 스텝)
                x_seq = ep_states[t - history_len + 1 : t + 1] # [H, 6]
                # t+1 시점의 목표 액션
                y_target = ep_actions[t + 1]                   # [6]
                
                self.samples_X.append(x_seq)
                self.samples_Y.append(y_target)
                
        self.samples_X = torch.tensor(np.array(self.samples_X), dtype=torch.float32)
        self.samples_Y = torch.tensor(np.array(self.samples_Y), dtype=torch.float32)
        
        self.history_len = history_len
        self.is_train = is_train
        
    def __len__(self):
        return len(self.samples_X)
        
    def __getitem__(self, idx):
        return self.samples_X[idx], self.samples_Y[idx]

def get_dataloaders(data_path="demonstrations.npz", history_len=10, batch_size=64):
    train_dataset = TimeSeriesBCDataset(data_path, history_len=history_len, is_train=True)
    test_dataset  = TimeSeriesBCDataset(data_path, history_len=history_len, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("=" * 60)
    print("📊 [Step 2] PyTorch 시계열 Dataset & DataLoader 검증")
    print("=" * 60)
    
    H = 10
    batch_size = 64
    train_loader, test_loader = get_dataloaders(history_len=H, batch_size=batch_size)
    
    print(f"  - Train Set 총 샘플 수: {len(train_loader.dataset)} 개 (80 에피소드 × 90 윈도우)")
    print(f"  - Test Set  총 샘플 수: {len(test_loader.dataset)} 개 (20 에피소드 × 90 윈도우)")
    print(f"  - 배치 크기(Batch Size) : {batch_size}")
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        print(f"\n🔍 [배치 텐서 차원(Shape) 검증]")
        print(f"  - 입력 텐서 (X) Shape: {batch_x.shape} (기대값: [Batch, History={H}, DoF=6])")
        print(f"  - 정답 텐서 (Y) Shape: {batch_y.shape} (기대값: [Batch, DoF=6])")
        break
        
    print("\n✅ Dataset & DataLoader 검증 성공!")
    print("=" * 60)
