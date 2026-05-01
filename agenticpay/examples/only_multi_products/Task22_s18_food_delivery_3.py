"""Task22 Scenario 18: Dripped Birria Sides - Two-Product Negotiation

Buyer negotiates for Dripped Nachos + Sprite.
Bundle purchase with delivered total price negotiation.
Category: Food Delivery
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
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance, OPENAI_API_KEY


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
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Use OpenAIVLM (Vision Language Model) for two-product negotiation with product images (image + text)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price are confidential reservation totals below product_info quoted_total_price / item reference sum.
    print("Creating agents...")
    product_request = (
        "I want Dripped Nachos and a Sprite from Dripped Birria. "
        "I also prefer the nachos cheese layer to look evenly melted rather than separated oily pools."
    )
    # Multi-dimensional contract setup for reusable MAUT scoring in env.
    contract_config = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No delivered total price, delivery speed, extra condiments option, or user product preference match has been selected or agreed "
                "before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, discrete_terms.delivery_speed, "
                "discrete_terms.extra_condiments, and discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": "The all-in delivered total amount the buyer pays for the whole two-item food delivery order, measured in US dollars.",
            "discrete_terms.delivery_speed": (
                "The delivery speed for this food order. `rush` means prioritized fastest delivery; "
                "`standard` means normal delivery; `batched` means slower bundled delivery with other orders."
            ),
            "discrete_terms.extra_condiments": (
                "Whether the restaurant includes extra condiments, sauces, or small sides with the order."
            ),
            "discrete_terms.user_product_preference": (
                "How well the menu imagery matches the buyer's stated preference for a nachos cheese layer that looks evenly melted "
                "rather than separated oily pools. Use `strong_match` when the preference is clearly satisfied, "
                "`partial_match` when it is only partly satisfied, and `mismatch_or_uncertain` when it is not "
                "satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {},
        "discrete_options": {
            "delivery_speed": ["rush", "standard", "batched"],
            "extra_condiments": [True, False],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 14.33,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for this delivered two-item food order before delivery speed and "
                    "condiment terms, measured in dollars. A lower price is better for you because every dollar "
                    "paid reduces your utility by 1 dollar."
                ),
                "discrete_weights.delivery_speed": (
                    "How much each delivery-speed option changes your utility, measured in dollars."
                ),
                "discrete_weights.extra_condiments": (
                    "How much the extra-condiments option changes your utility, measured in dollars."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated product preference changes your utility, "
                    "measured in dollars. Positive numbers are good for you; negative numbers are bad for you."
                ),
            },
            "continuous_weights": {},
            "discrete_weights": {
                "delivery_speed": {"rush": 3.0, "standard": 0.0, "batched": -2.0},
                "extra_condiments": {True: 1.5, False: 0.0},
                "user_product_preference": {
                    "strong_match": 0.30,
                    "partial_match": 0.12,
                    "mismatch_or_uncertain": -0.25,
                },
            },
        },
        "seller_preferences": {
            "c_base": 12.21,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost for fulfilling this delivered two-item food order before delivery "
                    "speed and condiment terms, measured in dollars. A higher price is better for you because every "
                    "dollar received increases your utility by 1 dollar."
                ),
                "discrete_weights.delivery_speed": (
                    "How much each delivery-speed option changes your utility, measured in dollars."
                ),
                "discrete_weights.extra_condiments": (
                    "How much the extra-condiments option changes your utility, measured in dollars."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated product preference changes your "
                    "utility, measured in dollars. Stronger commitments carry a small nonzero risk or handling cost."
                ),
            },
            "continuous_weights": {},
            "discrete_weights": {
                "delivery_speed": {"rush": -4.0, "standard": 0.0, "batched": 3.5},
                "extra_condiments": {True: -0.5, False: 0.0},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    buyer_max_price = contract_config["buyer_preferences"]["v_base"]  # Keep for backward-compatible price displays
    seller_min_price = contract_config["seller_preferences"]["c_base"]  # Keep for backward-compatible price displays
    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, name="Seller1", seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating two-product food delivery negotiation environment...")
    env = make(
        "Task2_two_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for both products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for both products)
        environment_info={
            "platform": "DoorDash",
            "market_type": "Food Delivery",
            "availability_status": "Available for delivery.",
            "restaurant_name": "Dripped Birria",
            "estimated_delivery_time": "20-30 minutes",
            "restaurant_price_range": "$$",
            "restaurant_address": "1731 Westheimer Rd, Houston, TX 77098, USA",
            "delivery_fee": 2.49,
            "service_fee": 1.21,
            "quoted_total_price": 17.70,
            "contract_config": contract_config,
        },
        price_tolerance=price_tolerance,
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    # Define two products from restaurantmenuchanges.csv (Dripped Birria)
    product_info = {
        "products": [
            {
                "name": "Dripped Nachos",
                "price": 9.50,
                "condition": "Prepared fresh to order",
                "brand": "Dripped Birria",
                "size": "Single shareable portion",
                "original_price": 9.50,
                "availability_status": "Available for delivery.",
                "product_category": "Food Delivery › Mexican › Sides",
                "average_rating": 4.31,
                "total_reviews": 557,
                "seller_name": "Dripped Birria",
                "asin": "DD-HOU-DRIPPED-BIRRIA-NACHOS",
                "full_description": "Nacho chips topped with cheese sauce, birria de res meat, jalapenos, onions, and cilantro. Menu data lists this item at $9.50.",
                "image_url": "https://img.cdn4dd.com/cdn-cgi/image/fit=contain,width=1200,height=672,format=auto/https://doordash-static.s3.amazonaws.com/media/photosV2/47d1ce98-66be-4bc9-9f0e-c41e08b60f4a-retina-large.jpg",
            },
            {
                "name": "Sprite",
                "price": 2.50,
                "condition": "Prepared fresh to order",
                "brand": "Dripped Birria",
                "original_price": 2.50,
                "availability_status": "Available for delivery.",
                "product_category": "Food Delivery › Mexican › Drinks",
                "average_rating": 4.31,
                "total_reviews": 557,
                "seller_name": "Dripped Birria",
                "asin": "DD-HOU-DRIPPED-BIRRIA-SPRITE",
                "full_description": "Canned Sprite drink listed on Dripped Birria menu at $2.50.",
                "image_url": "https://img.cdn4dd.com/cdn-cgi/image/fit=contain,width=1200,height=672,format=auto/https://doordash-static.s3.amazonaws.com/media/photosV2/ea8dea6a-4c27-4d1f-9c85-f0e2628ca0e5-retina-large.png",
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
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting two-product food delivery negotiation...")
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
        "task": "Task22_s18_food_delivery_3",
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
            if info.get('agreed_contract') is not None:
                print(f"Final Contract: {info['agreed_contract']}")
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
                "agreed_contract": info.get('agreed_contract'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "buyer_utility": info.get('buyer_utility'),
                "seller_utility": info.get('seller_utility'),
                "z_max": info.get('z_max'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
                "contract_config": contract_config,
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
        output_file = run_dir / "Task22_s18_food_delivery_3_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task22 Scenario 18: Dripped Birria Sides - Two-Product Negotiation Results\n")
            f.write("Category: Food Delivery\n")
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
            if results.get('agreed_contract') is not None:
                f.write(f"  Agreed Contract: {results['agreed_contract']}\n")
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
    parser = argparse.ArgumentParser(description="Task22 Scenario 18: Dripped Birria Sides - Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

