# syntax=docker/dockerfile:1.5
FROM vllm/vllm-openai:0.4.0

ENV HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    TRANSFORMERS_CACHE=/models

VOLUME ["/models"]
