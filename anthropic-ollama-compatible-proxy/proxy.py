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
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
from flask import Flask, request, jsonify, Response
from sseclient import SSEClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
FLASK_PORT = int(os.getenv('FLASK_PORT', 11434))

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

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
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        })

    def chat(self, messages: List[Message], options: ChatOptions, model: str) -> ChatResponse:
        """Send chat completion request to Claude API"""
        # Map Ollama model name to Claude model name
        claude_model = OLLAMA_TO_CLAUDE_MAPPING.get(model, model)

        # Separate system messages from chat messages
        system_messages = [m for m in messages if m.role == "system"]
        chat_messages = [m for m in messages if m.role != "system"]

        # Get system content (Claude API expects a single system parameter)
        system_content = system_messages[0].content if system_messages else ""

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

        logger.info(f"Sending request to Claude API with model: {claude_model}")

        try:
            response = self.session.post(
                f"{self.base_url}/messages",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            response_data = response.json()

            return ChatResponse(
                content=response_data["content"][0]["text"],
                model=claude_model,
                created_at=datetime.utcnow().isoformat() + "Z",
                prompt_tokens=response_data.get("usage", {}).get("input_tokens"),
                completion_tokens=response_data.get("usage", {}).get("output_tokens")
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Claude API response: {e}")
            raise

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models in Ollama format"""
        current_time = datetime.utcnow().isoformat() + "Z"

        models = []
        for ollama_name, claude_name in OLLAMA_TO_CLAUDE_MAPPING.items():
            # Determine model size based on model name
            if "opus" in claude_name.lower():
                size = 400_000_000_000
                parameter_size = "400B"
            else:  # sonnet
                size = 200_000_000_000
                parameter_size = "200B"

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

        return models

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        claude_model = OLLAMA_TO_CLAUDE_MAPPING.get(model_name, model_name)
        current_time = datetime.utcnow().isoformat() + "Z"

        if "opus" in claude_model.lower():
            description = "Most powerful model for highly complex tasks"
            parameter_size = "400B"
            parameter_count = 400_000_000_000
        else:  # sonnet
            description = "Our most intelligent model"
            parameter_size = "200B"
            parameter_count = 200_000_000_000

        return {
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


# Initialize Flask app and Claude provider
app = Flask(__name__)
claude_provider = ClaudeProvider(ANTHROPIC_API_KEY)


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return "Ollama is running"


@app.route('/api/tags', methods=['GET'])
def get_tags():
    """Get list of available models"""
    try:
        models = claude_provider.get_models()
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/show', methods=['POST'])
def show_model():
    """Show details about a specific model"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "Model name is required"}), 400

        model_name = data['name']
        details = claude_provider.get_model_details(model_name)
        return jsonify(details)

    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat completion requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Parse messages
        messages_data = data.get('messages', [])
        messages = [
            Message(role=msg.get('role', 'user'), content=msg.get('content', ''))
            for msg in messages_data
        ]

        # Parse options
        options_data = data.get('options', {})
        options = ChatOptions(
            temperature=float(options_data.get('temperature', 0.7)),
            top_p=float(options_data.get('top_p', 0.9)),
            max_tokens=options_data.get('max_tokens')
        )

        # Get model name
        model = data.get('model', 'claude-sonnet')

        # Get chat response from Claude
        response = claude_provider.chat(messages, options, model)

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

        return jsonify(ollama_response)

    except requests.exceptions.RequestException as e:
        logger.error(f"Claude API error: {e}")
        return jsonify({"error": f"Claude API error: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 - Path not found: {request.path}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 - Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info(f"Starting Ollama-to-Claude proxy server on port {FLASK_PORT}")
    logger.info(f"Available models: {list(OLLAMA_TO_CLAUDE_MAPPING.keys())}")

    app.run(
        host='0.0.0.0',
        port=FLASK_PORT,
        debug=False,
        threaded=True
    )