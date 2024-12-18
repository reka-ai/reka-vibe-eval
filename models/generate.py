"""
Main script to run vision models.

Usage:
    python main.py --model MODEL_NAME --data_path DATA_PATH [--api_key API_KEY] [--server_url SERVER_URL] [--server_port SERVER_PORT]

Examples
    # Run OpenAI model
    python main.py --model gpt-4-vision-preview --data_path data/vibe-eval.v1.jsonl

    # Run Gemini model
    python main.py --model gemini-pro-vision --data_path data/vibe-eval.v1.jsonl

    # Run Claude model
    python main.py --model claude-3-opus-20240229 --data_path data/vibe-eval.v1.jsonl

    # Run Pixtral server
    python main.py --model mistralai/Pixtral-Large-Instruct-2411 --server_url localhost --server_port 8000

    # Run X.AI model
    python main.py --model xai-vision --api_key YOUR_API_KEY
"""

import argparse
import os

def get_model(args):
    """Initialize the appropriate model based on arguments."""
    
    api_key = args.api_key

    # OpenAI models
    if args.model.startswith(("gpt-4", "o1")):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        from models.openai_models import OpenAIModel
        return OpenAIModel(args.model, api_key=api_key)
    
    # Gemini models
    elif args.model.startswith("gemini"):
        from models.gemini_models import GeminiModel
        return GeminiModel(args.model)
    
    # Claude models
    elif args.model.startswith("claude"):
        from models.claude_models import ClaudeModel
        return ClaudeModel(args.model)
    
    # Pixtral models
    elif "pixtral" in args.model.lower():
        if args.server_url:
            from models.pixtral_server import PixtralServer
            return PixtralServer(
                args.model,
                server_url=args.server_url,
                server_port=args.server_port
            )
        from models.pixtral_models import PixtralModel
        return PixtralModel(args.model)
    
    # X.AI models
    elif args.model.startswith("grok"):
        from models.xai_models import XAIModel
        return XAIModel(args.model, api_key=args.api_key)
    
    # Reka models
    elif args.model.startswith("reka"):
        from models.reka_models import RekaModel
        return RekaModel(args.model)
    
    else:
        raise ValueError(f"Unknown model: {args.model}")

def main():
    parser = argparse.ArgumentParser(description="Run vision models on evaluation data")
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/identifier"
    )
    parser.add_argument(
        "--data_path",
        default="data/vibe-eval.v1.jsonl",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key (required for some models)"
    )
    parser.add_argument(
        "--server_url",
        default="127.0.0.1",
        help="Server URL for Pixtral server"
    )
    parser.add_argument(
        "--server_port",
        default="8000",
        help="Server port for Pixtral server"
    )

    args = parser.parse_args()

    # Set API keys from environment if not provided
    if not args.api_key:
        if "OPENAI_API_KEY" in os.environ:
            args.api_key = os.environ["OPENAI_API_KEY"]
        elif "XAI_API_KEY" in os.environ:
            args.api_key = os.environ["XAI_API_KEY"]

    # Initialize model
    model = get_model(args)
    
    # Process examples
    model.process_examples(args.data_path)

if __name__ == "__main__":
    main() 