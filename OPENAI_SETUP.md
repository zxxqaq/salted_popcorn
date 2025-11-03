# OpenAI API 配置指南

## 1. 获取 OpenAI API Key

1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 登录你的账号
3. 点击右上角的头像 → "API keys"
4. 点击 "Create new secret key"
5. 复制生成的 API key（格式：`sk-...`）
   - ⚠️ **重要**：API key 只显示一次，请立即保存！

## 2. 配置 API Key

### 方法 1：使用环境变量（推荐）

**macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**永久设置（添加到 ~/.zshrc 或 ~/.bashrc）:**
```bash
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Windows (永久设置):**
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-key-here', 'User')
```

### 方法 2：在代码中直接设置

编辑 `src/vector_retrieval.py` 的 `__main__` 部分：

```python
if __name__ == "__main__":
    # 直接设置 API key（不推荐，但方便测试）
    api_key = "sk-your-key-here"  # 替换为你的 API key
    
    demonstrate_sample_query(
        project_root / "data" / "test",
        use_openai_api=True,
        api_base=None,  # None 表示使用标准 OpenAI API
        api_key=api_key,
    )
```

## 3. 配置说明

### 标准 OpenAI API 配置

代码已修改为默认使用标准 OpenAI API：
- **API Base URL**: `https://api.openai.com/v1`（自动）
- **认证方式**: `Authorization: Bearer {api_key}` header
- **模型**: `text-embedding-3-small` 或 `text-embedding-3-large`

### 如果需要使用自定义代理

如果你之后想使用其他代理，可以指定 `api_base`：

```python
demonstrate_sample_query(
    project_root / "data" / "test",
    use_openai_api=True,
    api_base="https://your-proxy-url.com/v1",
    api_key=api_key,
)
```

## 4. 测试配置

运行测试脚本：

```bash
cd /Users/xianzhang/Documents/prosus/Food-Semantic-Search
python3 src/vector_retrieval.py
```

如果配置正确，你应该看到：
- ✅ 成功加载 candidates
- ✅ 开始计算 embeddings
- ✅ 显示搜索结果

## 5. 价格和限制

### text-embedding-3-small
- **价格**: $0.02 per 1M tokens
- **最大输入**: 8,191 tokens per input
- **输出维度**: 1,536

### text-embedding-3-large
- **价格**: $0.13 per 1M tokens
- **最大输入**: 8,191 tokens per input
- **输出维度**: 3,072

### 速率限制
- 默认限制：**500 RPM** (Requests Per Minute)
- 可以在代码中通过 `rpm_limit` 参数调整

## 6. 注意事项

1. **API Key 安全**：
   - ⚠️ 不要将 API key 提交到 Git 仓库
   - ⚠️ 使用环境变量而不是硬编码
   - ⚠️ 定期轮换 API key

2. **成本控制**：
   - 代码已实现自动分批处理，避免单次请求过大
   - 监控 API 使用量：[OpenAI Usage Dashboard](https://platform.openai.com/usage)

3. **错误处理**：
   - 代码已包含错误处理和重试逻辑
   - 如果遇到 429（速率限制），会自动等待

## 7. 常见问题

**Q: 收到 401 认证错误？**
- 检查 API key 是否正确
- 检查 API key 是否过期
- 确认 API key 有足够的权限

**Q: 收到 429 速率限制错误？**
- 减少 `rpm_limit` 参数
- 增加批次之间的等待时间

**Q: 请求失败怎么办？**
- 检查网络连接
- 查看详细的错误日志
- 确认模型名称正确（`text-embedding-3-small` 或 `text-embedding-3-large`）

