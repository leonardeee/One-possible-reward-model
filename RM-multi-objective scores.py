import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型架构定义
class MultiObjectiveRewardModel(nn.Module):
    def __init__(self, base_model_name, num_objectives, hidden_size=1024):
        super(MultiObjectiveRewardModel, self).__init__()
        # 加载预训练的Llama3模型
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # 冻结所有层，避免在初始阶段更新这些权重
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 多目标奖励预测层
        self.reward_head = nn.Linear(self.base_model.config.hidden_size, num_objectives)
        
        # 门控层：多层感知机
        self.gating_layer = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_objectives),
            nn.Softmax(dim=-1)  # 确保输出为非负且和为1
        )
    
    def forward(self, input_ids):
        # 获取最后一个token的隐藏状态
        outputs = self.base_model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # 计算多目标奖励
        multi_obj_rewards = self.reward_head(last_hidden_state)
        
        # 计算门控层输出的权重
        gating_output = self.gating_layer(last_hidden_state)
        
        # 计算最终的加权奖励分数
        preference_score = (gating_output * multi_obj_rewards).sum(dim=-1)
        
        return multi_obj_rewards, gating_output, preference_score

# 数据准备
class RewardDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, response, rewards = self.data[idx]
        inputs = self.tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=self.max_length)
        return inputs.input_ids.squeeze(0), torch.tensor(rewards, dtype=torch.float)

# 数据集
train_data = [
    ("What is the capital of France?", "The capital of France is Paris.", [0.9, 0.8, 0.95]),
    ("Explain the theory of relativity.", "The theory of relativity was developed by Albert Einstein.", [0.85, 0.9, 0.92]),
    # 更多的数据样本
]

# 初始化和准备模型及数据
tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
train_dataset = RewardDataset(tokenizer, train_data)

# collate_fn用于处理批次中的输入
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    rewards = [item[1] for item in batch]
    
    # 对input_ids进行填充
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # 堆叠rewards
    rewards = torch.stack(rewards)
    
    return input_ids, rewards

# 修改DataLoader，增加collate_fn
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
num_objectives = 3  # 假设我们有三个奖励目标
model = MultiObjectiveRewardModel("sfairXC/FsfairX-LLaMA3-RM-v0.1", num_objectives).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

print(1)
# 模型训练
model.train()
for epoch in range(2):  # 设定10个epoch，可以根据需要调整
    total_loss = 0
    for input_ids, rewards in train_dataloader:
        input_ids, rewards = input_ids.to(device), rewards.to(device)
        
        optimizer.zero_grad()
        
        multi_obj_rewards, gating_output, preference_score = model(input_ids)
        
        # 计算损失
        loss = loss_fn(multi_obj_rewards, rewards)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

# 保存模型
# model.save_pretrained("E:/ntu-work")
# tokenizer.save_pretrained("E:/ntu-work")

# 保存模型权重
torch.save(model.state_dict(), 'E:/ntu-work/model_weights.pth')

# 保存tokenizer
tokenizer.save_pretrained('E:/ntu-work')
