"""Task12 Scenario 8: Jeans & Marvel T-Shirt Bundle - Two-Product Negotiation

Buyer negotiates for myhehthw Women's Jeans and Marvel Avengers T-Shirt bundle.
Bundle purchase with total price negotiation.
Category: Clothing & Fashion
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
    
    # OpenVLM via OpenAI-compatible API (product images passed to VLM)
    api_key = os.getenv("OPENAI_API_KEY") or "token-abc123"
    openvlm_base_url = os.getenv("OPENAI_URL") or os.getenv("OPENVLM_BASE_URL", "http://localhost:8000/v1")
    openvlm_model = os.getenv("OPENVLM_MODEL", "openvlm")
    
    model = OpenAIVLM(
        model=model_name or openvlm_model,
        api_key=api_key,
        base_url=openvlm_base_url,
    )

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total expected cost for both products
    print("Creating agents...")
    buyer_max_price = 40.0  # Maximum acceptable total purchase price for buyer (confidential) - Jeans $22 + T-Shirt $23
    seller_min_price = 30.0  # Minimum acceptable total selling price for seller (confidential) - Jeans $16 + T-Shirt $14
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating two-product negotiation environment...")
    env = make(
        "Task2_two_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_price=45.98,  # Initial total price offered by seller for both products ($22.99 + $22.99)
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for both products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for both products)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "availability_status": "In Stock",
        },
        price_tolerance=price_tolerance,
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Fashion-conscious customer looking for comfortable daily wear and casual Marvel fan apparel. Cares about fit, comfort, and value. Prefers stretchy denim and officially licensed merchandise."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices
    # Product 1: myhehthw Women's Jeans (from Task11 example)
    # Product 2: Marvel Avengers T-Shirt (from sampled_products2.jsonl line 8 / sample 8)
    product_info = {
        "products": [
            {
                "name": "myhehthw Women's High Waisted Jeans for Women Distressed Ripped Jeans Slim Fit Butt Lifting Skinny Stretch Jeans Trousers",
                "price": 22.99,
                "condition": "New",
                "brand": "myhehthw",
                "material": "75% Cotton, 20% Polyester, 5% Spandex",
                "sizes": ["Small", "Medium", "Large", "X-Large", "XX-Large"],
                "availability_status": "In Stock",
                "product_category": "Clothing, Shoes & Jewelry › Women › Clothing › Jeans",
                "asin": "B09PBNZLNT",
                "full_description": "Made of soft and stretchy denim cotton fabric, breathable to wear. Comfortable and versatile for casual, school, office, shopping. Button closure. Machine washable.",
                "image_url": "https://m.media-amazon.com/images/I/31Qo5LzPPjL.jpg",
            },
            {
                "name": "Marvel Avengers: Endgame Captain America America's Language T-Shirt",
                "price": 22.99,
                "condition": "New",
                "brand": "Brand: Marvel",
                "availability_status": "In Stock.",
                "product_category": "Clothing, Shoes & Jewelry › Novelty & More › Clothing › Novelty › Women › Tops & Tees › T-Shirts",
                "average_rating": 5,
                "total_reviews": 2,
                "seller_name": "",
                "asin": "B07XPR3R7N",
                "full_description": "Team up with what is left of the Avengers to fix the damage that Thanos has caused to the universe. You'll find the perfect gear within this collection of Officially Licensed Marvel Avengers: Endgame tee shirts, sweatshirts, and hoodies!",
                "image_url": "https://m.media-amazon.com/images/I/A13usaonutL.png",
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
    user_requirement = "I'm looking for a bundle: myhehthw Women's High Waisted Distressed Ripped Jeans, slim fit, size Medium or Large in black or dark wash, and a Marvel Avengers Captain America T-shirt in black, men's fit. Prefer comfortable and good value."
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
        "task": "Task12_s8_jeans_tshirt_bundle_negotiation",
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
        output_file = run_dir / "Task12_s8_jeans_tshirt_bundle_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task12 Scenario 8: Jeans & Marvel T-Shirt Bundle - Two-Product Negotiation Results\n")
            f.write("Category: Clothing & Fashion\n")
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
    parser = argparse.ArgumentParser(description="Task12 Scenario 8: Jeans & Marvel T-Shirt Bundle - Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)

