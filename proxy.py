#!/usr/bin/env python3
"""
Ollama API Proxy for Claude API

This application serves as a proxy that implements the Ollama API interface
but forwards requests to Anthropic's Claude API. This allows IDE plugins
that support Ollama to work with Claude models.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
from flask import Flask, request, jsonify, Response

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Create separate loggers for different components
api_logger = logging.getLogger('api_requests')
claude_logger = logging.getLogger('claude_client')
proxy_logger = logging.getLogger('proxy_server')

# Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
FLASK_PORT = int(os.getenv('FLASK_PORT', 11434))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Set log level from environment
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

proxy_logger.info(f"Starting proxy with log level: {LOG_LEVEL}")
proxy_logger.info(f"Flask port configured: {FLASK_PORT}")
proxy_logger.info(f"Anthropic API key configured: {'Yes' if ANTHROPIC_API_KEY else 'No'}")

# Model mappings from Ollama format to Claude format
OLLAMA_TO_CLAUDE_MAPPING = {
    'claude-4-sonnet': 'claude-4-sonnet-20250514',
    'claude-4-sonnet:latest': 'claude-4-sonnet-20250514',
    'claude-4-opus': 'claude-4-opus-20250514',
    'claude-4-opus:latest': 'claude-4-opus-20250514',
    'claude-sonnet': 'claude-4-sonnet-20250514',
    'claude-opus': 'claude-4-opus-20250514'
}

# Reverse mapping for model details
CLAUDE_TO_OLLAMA_MAPPING = {v: k for k, v in OLLAMA_TO_CLAUDE_MAPPING.items()}

proxy_logger.info(f"Model mappings loaded: {len(OLLAMA_TO_CLAUDE_MAPPING)} models available")
proxy_logger.debug(f"Available Ollama models: {list(OLLAMA_TO_CLAUDE_MAPPING.keys())}")


@dataclass
class Message:
    """Represents a chat message"""
    role: str
    content: str


@dataclass
class ChatOptions:
    """Chat completion options"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None


@dataclass
class ChatResponse:
    """Chat completion response"""
    content: str
    model: str
    created_at: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ClaudeProvider:
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str):
        claude_logger.info("Initializing Claude provider")
        self.api_key = api_key[:8] + "..." if api_key else None  # Log partial key for security
        self.base_url = "https://api.anthropic.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        })
        claude_logger.info(f"Claude provider initialized - Base URL: {self.base_url}")
        claude_logger.debug(f"API key (partial): {self.api_key}")

    def chat(self, messages: List[Message], options: ChatOptions, model: str) -> ChatResponse:
        """Send chat completion request to Claude API"""
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"

        claude_logger.info(f"[{request_id}] Starting chat request")
        claude_logger.info(f"[{request_id}] Input model: {model}")
        claude_logger.info(f"[{request_id}] Message count: {len(messages)}")
        claude_logger.info(
            f"[{request_id}] Options - temp: {options.temperature}, top_p: {options.top_p}, max_tokens: {options.max_tokens}")

        # Log message details (with content length for privacy)
        for i, msg in enumerate(messages):
            claude_logger.debug(f"[{request_id}] Message {i}: role={msg.role}, content_length={len(msg.content)}")
            if LOG_LEVEL == 'DEBUG':
                claude_logger.debug(f"[{request_id}] Message {i} content preview: {msg.content[:100]}...")

        # Map Ollama model name to Claude model name
        claude_model = OLLAMA_TO_CLAUDE_MAPPING.get(model, model)
        claude_logger.info(f"[{request_id}] Mapped to Claude model: {claude_model}")

        # Separate system messages from chat messages
        system_messages = [m for m in messages if m.role == "system"]
        chat_messages = [m for m in messages if m.role != "system"]

        claude_logger.debug(f"[{request_id}] System messages: {len(system_messages)}")
        claude_logger.debug(f"[{request_id}] Chat messages: {len(chat_messages)}")

        # Get system content (Claude API expects a single system parameter)
        system_content = system_messages[0].content if system_messages else ""
        if system_content:
            claude_logger.debug(f"[{request_id}] System content length: {len(system_content)}")

        # Prepare request payload
        payload = {
            "model": claude_model,
            "messages": [{"role": m.role, "content": m.content} for m in chat_messages],
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_tokens or 4096
        }

        if system_content:
            payload["system"] = system_content

        claude_logger.info(f"[{request_id}] Sending request to Claude API")
        claude_logger.debug(f"[{request_id}] Request payload keys: {list(payload.keys())}")

        # Log full payload in debug mode (be careful with sensitive content)
        if LOG_LEVEL == 'DEBUG':
            # Create safe payload for logging (truncate long content)
            safe_payload = payload.copy()
            if 'messages' in safe_payload:
                safe_payload['messages'] = [
                    {
                        'role': msg['role'],
                        'content': msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                    }
                    for msg in safe_payload['messages']
                ]
            if 'system' in safe_payload and len(safe_payload['system']) > 100:
                safe_payload['system'] = safe_payload['system'][:100] + '...'
            claude_logger.debug(f"[{request_id}] Payload preview: {json.dumps(safe_payload, indent=2)}")

        try:
            api_start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/messages",
                json=payload,
                timeout=60
            )
            api_duration = time.time() - api_start_time

            claude_logger.info(
                f"[{request_id}] Claude API response received - Status: {response.status_code}, Duration: {api_duration:.2f}s")
            claude_logger.debug(f"[{request_id}] Response headers: {dict(response.headers)}")

            response.raise_for_status()

            response_data = response.json()
            claude_logger.info(f"[{request_id}] Response parsed successfully")
            claude_logger.debug(f"[{request_id}] Response keys: {list(response_data.keys())}")

            # Log usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            claude_logger.info(f"[{request_id}] Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Log response content details
            content_blocks = response_data.get("content", [])
            claude_logger.info(f"[{request_id}] Content blocks: {len(content_blocks)}")

            if content_blocks:
                response_text = content_blocks[0].get("text", "")
                claude_logger.info(f"[{request_id}] Response text length: {len(response_text)}")
                claude_logger.debug(f"[{request_id}] Response preview: {response_text[:200]}...")

            total_duration = time.time() - start_time
            claude_logger.info(f"[{request_id}] Chat request completed - Total duration: {total_duration:.2f}s")

            return ChatResponse(
                content=content_blocks[0]["text"] if content_blocks else "",
                model=claude_model,
                created_at=datetime.utcnow().isoformat() + "Z",
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens
            )

        except requests.exceptions.Timeout as e:
            claude_logger.error(f"[{request_id}] Claude API timeout after {time.time() - start_time:.2f}s: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            claude_logger.error(f"[{request_id}] Claude API HTTP error - Status: {response.status_code}")
            try:
                error_data = response.json()
                claude_logger.error(f"[{request_id}] Error details: {json.dumps(error_data, indent=2)}")
            except:
                claude_logger.error(f"[{request_id}] Error response body: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            claude_logger.error(f"[{request_id}] Claude API request failed after {time.time() - start_time:.2f}s: {e}")
            raise
        except (KeyError, IndexError) as e:
            claude_logger.error(f"[{request_id}] Failed to parse Claude API response: {e}")
            claude_logger.error(f"[{request_id}] Response data: {json.dumps(response_data, indent=2)}")
            raise

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models in Ollama format"""
        claude_logger.info("Generating models list")
        current_time = datetime.utcnow().isoformat() + "Z"

        models = []
        for ollama_name, claude_name in OLLAMA_TO_CLAUDE_MAPPING.items():
            claude_logger.debug(f"Processing model mapping: {ollama_name} -> {claude_name}")

            # Determine model size based on model name
            if "opus" in claude_name.lower():
                size = 400_000_000_000
                parameter_size = "400B"
                claude_logger.debug(f"Model {ollama_name} classified as Opus (400B)")
            else:  # sonnet
                size = 200_000_000_000
                parameter_size = "200B"
                claude_logger.debug(f"Model {ollama_name} classified as Sonnet (200B)")

            model_info = {
                "name": ollama_name,
                "model": ollama_name,
                "modified_at": current_time,
                "size": size,
                "digest": f"anthropic-{claude_name}",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "claude",
                    "families": ["claude"],
                    "parameter_size": parameter_size,
                    "quantization_level": "Q4_K_M"
                }
            }
            models.append(model_info)

        claude_logger.info(f"Generated {len(models)} model entries")
        return models

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        claude_logger.info(f"Getting model details for: {model_name}")

        claude_model = OLLAMA_TO_CLAUDE_MAPPING.get(model_name, model_name)
        claude_logger.debug(f"Mapped model name: {model_name} -> {claude_model}")

        current_time = datetime.utcnow().isoformat() + "Z"

        if "opus" in claude_model.lower():
            description = "Most powerful model for highly complex tasks"
            parameter_size = "400B"
            parameter_count = 400_000_000_000
            claude_logger.debug(f"Model {model_name} configured as Opus")
        else:  # sonnet
            description = "Our most intelligent model"
            parameter_size = "200B"
            parameter_count = 200_000_000_000
            claude_logger.debug(f"Model {model_name} configured as Sonnet")

        details = {
            "license": "Anthropic Research License",
            "system": description,
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": parameter_size,
                "quantization_level": "Q4_K_M"
            },
            "model_info": {
                "general.architecture": "claude",
                "general.file_type": 15,
                "general.context_length": 200000,
                "general.parameter_count": parameter_count
            },
            "modified_at": current_time
        }

        claude_logger.info(f"Model details generated for {model_name}")
        return details


# Initialize Flask app and Claude provider
app = Flask(__name__)
claude_provider = ClaudeProvider(ANTHROPIC_API_KEY)

proxy_logger.info("Flask app and Claude provider initialized")


@app.before_request
def log_request_info():
    """Log incoming request details"""
    api_logger.info(f"Incoming request: {request.method} {request.path}")
    api_logger.debug(f"Request headers: {dict(request.headers)}")
    api_logger.debug(f"Request remote addr: {request.remote_addr}")
    api_logger.debug(f"Request user agent: {request.headers.get('User-Agent', 'Unknown')}")

    if request.content_type and 'application/json' in request.content_type:
        try:
            if request.get_json():
                data = request.get_json()
                # Log request data safely (avoid logging sensitive content)
                safe_data = {}
                for key, value in data.items():
                    if key == 'messages' and isinstance(value, list):
                        safe_data[key] = f"[{len(value)} messages]"
                    elif isinstance(value, str) and len(value) > 100:
                        safe_data[key] = f"{value[:50]}... (truncated, length: {len(value)})"
                    else:
                        safe_data[key] = value
                api_logger.debug(f"Request JSON: {json.dumps(safe_data, indent=2)}")
        except Exception as e:
            api_logger.debug(f"Could not parse request JSON: {e}")


@app.after_request
def log_response_info(response):
    """Log response details"""
    api_logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
    api_logger.debug(f"Response headers: {dict(response.headers)}")

    if response.content_type and 'application/json' in response.content_type:
        try:
            # Only log response data in debug mode and limit size
            if LOG_LEVEL == 'DEBUG' and response.data:
                response_text = response.get_data(as_text=True)
                if len(response_text) > 500:
                    api_logger.debug(f"Response JSON (truncated): {response_text[:500]}...")
                else:
                    api_logger.debug(f"Response JSON: {response_text}")
        except Exception as e:
            api_logger.debug(f"Could not log response JSON: {e}")

    return response


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_logger.info("Health check requested")
    return "Ollama is running"


@app.route('/api/tags', methods=['GET'])
def get_tags():
    """Get list of available models"""
    try:
        api_logger.info("Models list requested")
        start_time = time.time()

        models = claude_provider.get_models()

        duration = time.time() - start_time
        api_logger.info(f"Models list generated successfully in {duration:.3f}s - {len(models)} models")

        return jsonify({"models": models})
    except Exception as e:
        api_logger.error(f"Error getting models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/show', methods=['POST'])
def show_model():
    """Show details about a specific model"""
    try:
        data = request.get_json()
        api_logger.info("Model details requested")

        if not data or 'name' not in data:
            api_logger.warning("Model details request missing model name")
            return jsonify({"error": "Model name is required"}), 400

        model_name = data['name']
        api_logger.info(f"Model details requested for: {model_name}")

        start_time = time.time()
        details = claude_provider.get_model_details(model_name)
        duration = time.time() - start_time

        api_logger.info(f"Model details generated for {model_name} in {duration:.3f}s")
        return jsonify(details)

    except Exception as e:
        api_logger.error(f"Error getting model details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat completion requests"""
    request_start_time = time.time()
    request_id = f"chat_{int(request_start_time * 1000)}"

    try:
        api_logger.info(f"[{request_id}] Chat request received")

        data = request.get_json()
        if not data:
            api_logger.warning(f"[{request_id}] Chat request missing body")
            return jsonify({"error": "Request body is required"}), 400

        # Parse and log request parameters
        messages_data = data.get('messages', [])
        model = data.get('model', 'claude-sonnet')
        options_data = data.get('options', {})

        api_logger.info(f"[{request_id}] Request params - model: {model}, messages: {len(messages_data)}")
        api_logger.debug(f"[{request_id}] Raw options: {options_data}")

        # Parse messages
        messages = [
            Message(role=msg.get('role', 'user'), content=msg.get('content', ''))
            for msg in messages_data
        ]

        # Log message summary
        role_counts = {}
        total_content_length = 0
        for msg in messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
            total_content_length += len(msg.content)

        api_logger.info(f"[{request_id}] Message breakdown: {role_counts}, total content: {total_content_length} chars")

        # Parse options
        options = ChatOptions(
            temperature=float(options_data.get('temperature', 0.7)),
            top_p=float(options_data.get('top_p', 0.9)),
            max_tokens=options_data.get('max_tokens')
        )

        api_logger.debug(
            f"[{request_id}] Parsed options: temp={options.temperature}, top_p={options.top_p}, max_tokens={options.max_tokens}")

        # Get chat response from Claude
        api_logger.info(f"[{request_id}] Forwarding to Claude API")
        claude_start_time = time.time()

        response = claude_provider.chat(messages, options, model)

        claude_duration = time.time() - claude_start_time
        api_logger.info(f"[{request_id}] Claude response received in {claude_duration:.2f}s")

        # Format response in Ollama format
        ollama_response = {
            "model": model,  # Return the original Ollama model name
            "created_at": response.created_at,
            "message": {
                "role": "assistant",
                "content": response.content
            },
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": response.prompt_tokens,
            "eval_count": response.completion_tokens,
            "eval_duration": 0
        }

        total_duration = time.time() - request_start_time
        api_logger.info(f"[{request_id}] Chat request completed successfully")
        api_logger.info(
            f"[{request_id}] Total duration: {total_duration:.2f}s, Claude duration: {claude_duration:.2f}s")
        api_logger.info(f"[{request_id}] Response length: {len(response.content)} chars")
        api_logger.info(
            f"[{request_id}] Token usage - Input: {response.prompt_tokens}, Output: {response.completion_tokens}")

        return jsonify(ollama_response)

    except requests.exceptions.Timeout as e:
        duration = time.time() - request_start_time
        api_logger.error(f"[{request_id}] Request timeout after {duration:.2f}s: {e}")
        return jsonify({"error": f"Request timeout: {str(e)}"}), 504
    except requests.exceptions.HTTPError as e:
        duration = time.time() - request_start_time
        api_logger.error(f"[{request_id}] Claude API HTTP error after {duration:.2f}s: {e}")
        return jsonify({"error": f"Claude API error: {str(e)}"}), 502
    except requests.exceptions.RequestException as e:
        duration = time.time() - request_start_time
        api_logger.error(f"[{request_id}] Claude API request failed after {duration:.2f}s: {e}")
        return jsonify({"error": f"Claude API error: {str(e)}"}), 502
    except Exception as e:
        duration = time.time() - request_start_time
        api_logger.error(f"[{request_id}] Chat error after {duration:.2f}s: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    api_logger.warning(f"404 - Path not found: {request.path} from {request.remote_addr}")
    api_logger.debug(
        f"404 - Request method: {request.method}, User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    api_logger.error(f"500 - Internal server error: {error}", exc_info=True)
    api_logger.error(f"500 - Request: {request.method} {request.path} from {request.remote_addr}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    proxy_logger.info("=" * 60)
    proxy_logger.info("STARTING OLLAMA-TO-CLAUDE PROXY SERVER")
    proxy_logger.info("=" * 60)
    proxy_logger.info(f"Server configuration:")
    proxy_logger.info(f"  - Port: {FLASK_PORT}")
    proxy_logger.info(f"  - Log level: {LOG_LEVEL}")
    proxy_logger.info(f"  - Available models: {len(OLLAMA_TO_CLAUDE_MAPPING)}")
    proxy_logger.info(f"  - Model list: {', '.join(OLLAMA_TO_CLAUDE_MAPPING.keys())}")
    proxy_logger.info(f"  - Claude API: {'Connected' if ANTHROPIC_API_KEY else 'Not configured'}")
    proxy_logger.info("=" * 60)

    try:
        app.run(
            host='0.0.0.0',
            port=FLASK_PORT,
            debug=False,
            threaded=True
        )
    except Exception as e:
        proxy_logger.error(f"Failed to start server: {e}", exc_info=True)
        raise