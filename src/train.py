import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Sampler
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score, log_loss
from src.data_utils import aggregate_daily

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-BCE_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class DailyBatchSampler(Sampler):
    def __init__(self, day_ids, num_days_per_batch=10):
        self.day_ids = day_ids.cpu().numpy() if torch.is_tensor(day_ids) else np.array(day_ids)
        self.num_days_per_batch = num_days_per_batch
        
        # map each unique day to the list of sample indices belonging to that day
        self.day_to_indices = {}
        for idx, day in enumerate(self.day_ids):
            if day not in self.day_to_indices:
                self.day_to_indices[day] = []
            self.day_to_indices[day].append(idx)
        
        self.unique_days = sorted(list(self.day_to_indices.keys()))

    def __iter__(self):
        # process days in order to maintain temporal structure
        for i in range(0, len(self.unique_days), self.num_days_per_batch):
            batch_indices = []
            selected_days = self.unique_days[i : i + self.num_days_per_batch]
            for day in selected_days:
                batch_indices.extend(self.day_to_indices[day])
            yield batch_indices

    def __len__(self):
        return (len(self.unique_days) + self.num_days_per_batch - 1) // self.num_days_per_batch

def get_daily_pos_weight(y_train, tr_day_ids):
    
    unique_days = torch.unique(tr_day_ids)
    daily_targets = torch.zeros(len(unique_days), device=DEVICE)
    daily_targets.scatter_(0, tr_day_ids, y_train)
    
    count_calm = (daily_targets == 0).sum().float()
    count_fragile = (daily_targets == 1).sum().float()
    
    return torch.tensor([count_calm / count_fragile], device=DEVICE)

def train_nn_model(model, train_data, val_data, tr_indices, va_indices, train_df, val_df, lr, epochs=50, patience=5, alpha=0.7, gamma=2):
    X_train, y_train = torch.FloatTensor(train_data[0]).to(DEVICE), torch.FloatTensor(train_data[1]).to(DEVICE)
    X_val, y_val = torch.FloatTensor(val_data[0]).to(DEVICE), torch.FloatTensor(val_data[1]).to(DEVICE)
    
    def get_day_ids(df, indices):
        dates = df.loc[indices, 'Date'].values
        _, inverse = np.unique(dates, return_inverse=True)
        return torch.LongTensor(inverse).to(DEVICE)

    tr_day_ids = get_day_ids(train_df, tr_indices)
    va_day_ids = get_day_ids(val_df, va_indices)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, tr_day_ids), 
        batch_sampler=DailyBatchSampler(tr_day_ids, num_days_per_batch=21)
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, va_day_ids), 
        batch_sampler=DailyBatchSampler(va_day_ids, num_days_per_batch=42)
    )
    
    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc, trigger = 0, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_b, y_b, day_ids_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b).squeeze()
            
            unique_days_b, batch_inv = torch.unique(day_ids_b, return_inverse=True)
            daily_logits_sum = torch.zeros(len(unique_days_b), device=DEVICE).scatter_add(0, batch_inv, logits)
            daily_counts = torch.zeros(len(unique_days_b), device=DEVICE).scatter_add(0, batch_inv, torch.ones_like(logits))
            daily_mean_logits = daily_logits_sum / daily_counts
            
            batch_daily_targets = torch.zeros(len(unique_days_b), device=DEVICE).scatter_(0, batch_inv, y_b)
            
            loss = criterion(daily_mean_logits, batch_daily_targets)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        v_probs, v_targets, v_days = [], [], []
        with torch.no_grad():
            for X_v, y_v, day_ids_v in val_loader:
                v_logits = model(X_v).squeeze()
                if v_logits.dim() == 0: v_logits = v_logits.unsqueeze(0)
                v_probs.extend(torch.sigmoid(v_logits).cpu().numpy())
                v_targets.extend(y_v.cpu().numpy())
                v_days.extend(day_ids_v.cpu().numpy())
        
        unique_val_dates = np.sort(val_df['Date'].unique())
        aligned_dates = unique_val_dates[v_days] 
        
        eval_df = pd.DataFrame({
            'Date': aligned_dates,
            'Target': v_targets,
            'Prob': v_probs
        })
        
        daily = eval_df.groupby('Date').agg({'Target': 'first', 'Prob': 'mean'})
        y_true_daily = daily['Target'].values
        y_probs_daily = daily['Prob'].values
        
        # technically not needed after decreasing number of splits
        if len(np.unique(y_true_daily)) < 2:
            val_auc = 0.5 
        else:
            val_auc = roc_auc_score(y_true_daily, y_probs_daily)

        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience: break
            
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc

def train_student(model_name, model_configs, teacher_soft_labels, data_dict, df, test_indices):
    config = model_configs[model_name]
    model_cls = config['model_class']
    params = config['params']
    
    student_model = model_cls(**params)

    student_model.fit(
        data_dict['train']['beh'], 
        teacher_soft_labels, 
    )
    
    train_probs = student_model.predict(data_dict['train']['beh'])
    y_train_true_daily, y_train_probs_daily = aggregate_daily(
        df, 
        data_dict['train']['indices'], 
        data_dict['train']['target'], 
        train_probs
    )
    train_auc = roc_auc_score(y_train_true_daily, y_train_probs_daily)

    test_probs = student_model.predict(data_dict['test']['beh'])
    y_test_true_daily, y_test_probs_daily = aggregate_daily(
        df, 
        test_indices, 
        data_dict['test']['target'], 
        test_probs
    )

    test_auc = roc_auc_score(y_test_true_daily, y_test_probs_daily)
    test_loss = log_loss(y_test_true_daily, y_test_probs_daily)

    print(f"Student ({model_name}) Performance:")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Log Loss: {test_loss:.4f}")
    
    metrics = {'auc': test_auc, 'loss': test_loss}
    return student_model, metrics