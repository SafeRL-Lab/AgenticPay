"""Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Two-Product Negotiation

Buyer negotiates for Kids Wireless Headphones and Sony Bluetooth Speaker bundle.
Bundle purchase with total price negotiation.
Category: Electronics
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project path (4 levels up from script to reach repo root AgenticPayGym)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay import make  # Use registration system
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def get_model_name(model):
    """Extract model name from model object
    
    Args:
        model: Model object (CustomLLM, VLLMLLM, etc.)
    
    Returns:
        str: Model name
    """
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'model_id'):
        return model.model_id
    elif hasattr(model, 'model_path'):
        # Extract model name from path
        model_path = model.model_path
        return os.path.basename(model_path) if model_path else str(model)
    else:
        # Fallback to string representation, but try to extract model name
        model_str = str(model)
        # Try to extract model name from string like "CustomLLM(model=qwen3-8b)"
        if "model=" in model_str:
            try:
                return model_str.split("model=")[1].split(")")[0]
            except:
                return model_str
        else:
            return model_str


def main(model_name=None):
    """Main function: Demonstrates two-product negotiation flow
    
    Args:
        model_name: Optional model name. If None, uses default model.
    """
    
    print("Initializing model...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Use OpenAIVLM (Vision Language Model) for bundle negotiation with product images (图文)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

    # Alternative: CustomLLM for text-only models
    # model = CustomLLM(api_key=api_key, model=model_name)

    # Build absolute path to model directory
    # model_path = os.path.join(project_root, "models", "download_models", "Qwen3-8B-Instruct")
    # model_path = os.path.abspath(model_path)

    # vLLM LLM Model
    # model = VLLMLLM(
    #     model_path=model_path,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=4, # 4 GPUs
    # )

    # SGLang VLM Model
    # model = SGLangVLM(
    #     model_path=model_path,
    # )

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total expected cost for both products
    print("Creating agents...")
    buyer_max_price = 130.0  # Maximum acceptable total purchase price for buyer (confidential) - Headphones + Speaker bundle
    seller_min_price = 95.0  # Minimum acceptable total selling price for seller (confidential) - Headphones $10 + Speaker $85
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating two-product negotiation environment...")
    env = make(
        "Task2_two_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_price=123.48,  # Initial total price offered by seller for both products ($14.99 + $108.49)
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for both products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for both products)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "availability_status": "In Stock.",
        },
        price_tolerance=price_tolerance,
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Parent looking for affordable kids headphones for school and travel, plus a portable Bluetooth speaker for outdoor activities. Values volume control for child safety, durability, good battery life, and good value."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices (Product 1 from Task7 example, Product 2 from sampled_products2.jsonl 4th)
    product_info = {
        "products": [
            {
                "name": "Kids Wireless Headphones, Adjustable Headband, Stereo Sound, 3.5mm Jack, Kids Bluetooth Headphones, Volume Control, Foldable, Build-in Microphone, Over-Ear Headphones for Kids for School Home, Travel",
                "price": 14.99,
                "condition": "New",
                "brand": "Brand: NVRADCHUA",
                "availability_status": "In Stock.",
                "product_category": "Electronics › Headphones › Over-Ear Headphones",
                "average_rating": 4.0,
                "total_reviews": 2,
                "seller_name": "Manyutech",
                "asin": "B09KQNH5C6",
                "full_description": "WIRELESS & WIRED KIDS HEADPHONES: Built with 5.0 Bluetooth chip for fast and stable connection, also with 3.5mm jack. Compatible with smartphones, laptops, tablets, computers, TVs. Cute cat headphones designed with cartoon pattern, comfortable and soft ear cushions to protect child's ears. Excellent sound quality and adjustable headband, stretchable and foldable design for travel and storage. Long battery life (up to 7 hours) and built-in microphone for calls, video chats, or online lessons. Perfect for kids headphones for school and outdoor use.",
                "image_url": "https://m.media-amazon.com/images/I/41B+OC0qnOL.jpg",
            },
            {
                "name": "Sony Extra Bass Portable Bluetooth Speaker Black - SRS-XB33/BC (Renewed)",
                "price": 108.49,
                "condition": "Renewed",
                "brand": "Visit the Amazon Renewed Store",
                "availability_status": "Only 1 left in stock - order soon.",
                "product_category": "Electronics › Portable Audio & Video › Portable Bluetooth Speakers",
                "average_rating": 4.5,
                "total_reviews": 962,
                "seller_name": "Planet Open Box",
                "asin": "B08FZDJRQ7",
                "full_description": "This pre-owned or refurbished product has been professionally inspected and tested to work and look like new. How a product becomes part of Amazon Renewed, your destination for pre-owned, refurbished products: A customer buys a new product and returns it or trades it in for a newer or different model. That product is inspected and tested to work and look like new by Amazon-qualified suppliers. Then, the product is sold as an Amazon Renewed product on Amazon. If not satisfied with the purchase, renewed products are eligible for replacement or refund under the Amazon Renewed Guarantee.",
                "image_url": "https://m.media-amazon.com/images/I/41+lMIUpYbL.jpg",
                "small_description": ["Play it loud with EXTRA BASS sound and LIVE SOUND mode", "Built to last with a IP67 waterproof rustproof dustproof and shockproof design", "Lasts all day long with up to 24 hours of battery life", "X-Balanced Speaker Unit enhances sound quality and power", "Get things booming with Party Connect and sync up to 100 speakers", "Wireless with BLUETOOTH technology and NFC"],
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts:")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Product Price: ${total_product_price:.2f}")
    
    # Get user requirement (should describe purchasing two products)
    # Use default requirement for automatic running
    user_requirement = "I'm looking for kids wireless headphones with volume control, Bluetooth and 3.5mm jack for school and travel, plus a portable Bluetooth speaker with good bass and waterproof design for outdoor use. I want to buy both as a bundle."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting two-product negotiation...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task8_s4_headphones_speaker_bundle_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round: buyer responds first, then seller responds (seeing buyer's message)
        # Get buyer's response
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = observation["conversation_history"].copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
            current_state=observation
        )
        
        # Execute step with both actions
        observation, reward, terminated, truncated, info = env.step(
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # If this is the final round (agreed or timeout), display score calculations after Round Summary
        if done:
            # Print score calculations after Round Summary
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            seller_price = info.get('seller_price')
            buyer_price = info.get('buyer_price')
            seller_price_str = f"${seller_price:.2f}" if seller_price is not None else "Not specified"
            buyer_price_str = f"${buyer_price:.2f}" if buyer_price is not None else "Not specified"
            print(f"Final Total Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
            if info.get('agreed_price'):
                print(f"Agreed Total Price: ${info.get('agreed_price', 0):.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Reward: {reward:.3f}")
            if 'global_score' in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if 'buyer_score' in info:
                print(f"BuyerScore: {info['buyer_score']:.3f}")
            if 'seller_score' in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            
            # Collect results
            elapsed_time = time.time() - start_time
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nTwo-product negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_products"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name for directory (sanitize for filesystem)
        model_name = get_model_name(model)
        model_name_safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save output text
        output_file = run_dir / "Task8_s4_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Two-Product Negotiation Results\n")
            f.write("Category: Electronics\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            elapsed_time = results.get('elapsed_time', 0)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller Total Price: ${results['seller_price']:.2f}" if results.get('seller_price') is not None else "  Seller Total Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer Total Price: ${results['buyer_price']:.2f}" if results.get('buyer_price') is not None else "  Buyer Total Price: Not specified")
            f.write("\n")
            if results.get('agreed_price'):
                f.write(f"  Agreed Total Price: ${results['agreed_price']:.2f}\n")
            f.write("\n")
            f.write("Products:\n")
            for i, p in enumerate(results.get('product_info', {}).get('products', []), 1):
                f.write(f"  {i}. {p.get('name', 'Unknown')}: ${p.get('price', 0):.2f}\n")
            f.write("\n")
            f.write("Scores:\n")
            if results.get('global_score') is not None:
                f.write(f"  Global Score: {results['global_score']:.3f}\n")
            if results.get('buyer_score') is not None:
                f.write(f"  Buyer Score: {results['buyer_score']:.3f}\n")
            if results.get('seller_score') is not None:
                f.write(f"  Seller Score: {results['seller_score']:.3f}\n")
            f.write("\n")
            if results.get('termination_reason'):
                f.write(f"Termination Reason: {results['termination_reason']}\n")
            if results.get('error'):
                f.write(f"\nError: {results['error']}\n")
        
        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

