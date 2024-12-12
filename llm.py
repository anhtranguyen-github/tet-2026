from transformers import pipeline
model_path = '5CD-AI/Vietnamese-Sentiment-visobert'

# Pass `device=0` to use the GPU
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=0)

# Perform sentiment analysis
result = sentiment_task("Tụi mày tỉnh táo đi. Nếu là họ bị đào mồ cuốc mà cả 2-3 tháng nay. Họ tức _ tức run người chứ sao. Ai mà k tức")
print(result)
