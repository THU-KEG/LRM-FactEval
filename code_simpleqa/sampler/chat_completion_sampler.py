import base64
import time
import json
import os
from typing import Any

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..types import MessageList, SamplerBase

# 添加全局错误日志文件路径
ERROR_LOG_DIR = "error_logs"


def log_error_to_file(error_data,GLOBAL_ERROR_LOG):
    """记录错误到全局日志文件"""
    try:
        # 读取现有记录
        with open(GLOBAL_ERROR_LOG, 'r') as f:
            errors = json.load(f)
        
        # 添加新记录
        errors.append(error_data)
        
        # 写回文件
        with open(GLOBAL_ERROR_LOG, 'w') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
            
        print(f"错误已记录: {error_data['error_type']} - {error_data['timestamp']}")
    except Exception as e:
        print(f"记录错误时失败: {str(e)}")

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        model_name: str = "",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 32768,
        url: str = "",
        use_chat: bool = True,  # 添加是否使用chat模式的标识
        max_retries: int = 3,
        timeout: int = 360,  # 增加超时时间到120秒
        enable_thinking: bool = True,
        model_config: str = ""
    ):
        
        self.client = OpenAI(
            base_url=url,  # 替换为你的服务器IP和端口
            api_key=""  # 随便设置一个值，因为我们的服务不验证key
        )
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.use_chat = use_chat  # 保存是否使用chat模式
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_thinking = enable_thinking # 保存超时时间
        self.model_config = model_config
        self.model_name = model_name
        os.makedirs(ERROR_LOG_DIR, exist_ok=True)
        self.GLOBAL_ERROR_LOG = os.path.join(ERROR_LOG_DIR, f"all_errors_{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json")

        # 初始化错误日志文件
        if not os.path.exists(self.GLOBAL_ERROR_LOG):
            with open(self.GLOBAL_ERROR_LOG, 'w') as f:
                json.dump([], f)
            print(f"全局错误日志文件创建于: {self.GLOBAL_ERROR_LOG}")
    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    @retry(
        stop=stop_after_attempt(5),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=4, max=30),  # 指数退避重试
        reraise=False  # 修改为不重新抛出异常
    )
    def __call__(self, message_list: MessageList) -> str:
        # 提取prompt内容用于错误记录
        prompt_content = ""
        if message_list and len(message_list) > 0:
            prompt_content = message_list[0]
        
        try:
            if self.use_chat:
                # 使用chat.completions（用于grader）
                if self.enable_thinking:
                    params = {
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking},"top_k":20},
                        "temperature": 0.6 if self.enable_thinking else 0.7,
                        "top_p": 0.95 if self.enable_thinking else 0.8,
                    }
                else:
                    if self.model_config == "mimo":
                        params = {
                        "temperature": 0.6,
                        # "top_p": 0.8,
                    }
                    elif self.model_config == "dapo":
                        params = {
                            "temperature": 1,
                            "top_p": 0.7,
                            
                        }
                    elif self.model_config == "qwen":
                        params = {
                            "temperature": 0.7,
                            "top_p": 0.8,
                        }
                    elif self.model_config == "distill":
                        params = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                        }
                    elif self.model_config == "deepseek-r1":
                        params = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                        }
                    elif self.model_config == "deepseek-v3":
                        params = {
                            
                        }
                    elif self.model_config == "qwen3":
                        params = {
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking},"top_k":20},
                        "temperature": 0.7,
                        "top_p": 0.8,
                    }
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    **params
                )
                return response.choices[0].message.content
            else:
                if self.model_config == "mimo":
                    params = {
                        "temperature": 0.6,
                        #"top_p": 0.8,
                    }
                elif self.model_config == "dapo":
                    params = {
                        "temperature": 1,
                        "top_p": 0.7,
                    }
                elif self.model_config == "qwen":
                    params = {
                        "temperature": 0.7,
                        "top_p": 0.8,
                    }
                elif self.model_config == "deepseek-v3":
                    params = {
                        
                    }
                elif self.model_config == "deepseek-r1":
                    params = {
                        "temperature": 0.6,
                        "top_p": 0.95,
                    }
                # 使用completions（用于model）
                prompt = message_list[0]['content']
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    **params
                )
                return response.choices[0].text
        except Exception as e:
            error_msg = str(e)
            print(f"Error after all retries: {error_msg}")
            
            # 记录错误到全局日志文件
            error_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model,
                "model_name": self.model_name,
                "error_type": str(type(e).__name__),
                "error_message": error_msg,
                "context": {
                    "prompt_excerpt": prompt_content,
                    "timeout_setting": self.timeout,
                    "max_tokens": self.max_tokens
                }
            }
            
            log_error_to_file(error_data,self.GLOBAL_ERROR_LOG)
            
            # 返回带有完整错误信息的错误消息
            return f"ERROR: {error_msg}"
