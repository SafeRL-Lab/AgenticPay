"""Task22 Scenario 19: Food Delivery (Dripped Nachos) Negotiation

Category: Food Delivery
Scenario: Dripped Nachos order from Dripped Birria on DoorDash.
Tests agent's ability to negotiate over a restaurant menu item with delivery context.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add project path
# Script is at: agenticpay/examples/single_buyer_product_seller/Task1_basic_price_negotiation.py
# Need to go up 4 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)   

from agenticpay import make, Task1BasicPriceNegotiation  # Use registration system
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM

from agenticpay.examples.config import reward_weights, max_rounds, price_tolerance, OPENAI_API_KEY


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
    """Main function: Demonstrates basic negotiation flow
    
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
    
    # Use OpenAIVLM (Vision Language Model) for product negotiation with product images
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

    # Alternative: CustomLLM for text-only models
    # model = CustomLLM(api_key=api_key, model=model_name)

    # Alternative: SGLang VLM (local)
    # model_path = os.path.join(project_root, "models", "download_models", "Qwen3-VL-8B-Instruct")
    # model = SGLangVLM(model_path=os.path.abspath(model_path))

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    # Scenario 19: Dripped Nachos — public all-in reference (quoted_total_price $13.14); confidential walk-aways sit well below that anchor.
    contract_config = {
        "contrainfo": {
            "product_request": (
                "I want Dripped Nachos from Dripped Birria, delivered. "
                "I also prefer the chips under the toppings to still look mostly crisp, not fully soaked and collapsed."
            ),
            "initial_contract_status": (
                "No all-in price, delivery speed, extra condiments option, or user product preference match has been "
                "selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, discrete_terms.delivery_speed, "
                "discrete_terms.extra_condiments, and discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": (
                "The all-in order total the buyer pays, measured in US dollars, including the food item "
                "and mandatory delivery-related fees."
            ),
            "discrete_terms.delivery_speed": (
                "The fulfillment speed for the DoorDash order. `rush` means prioritized delivery; "
                "`standard` means normal delivery; `batched` means slower grouped delivery."
            ),
            "discrete_terms.extra_condiments": (
                "Whether the order includes extra condiments, sauces, or small sides requested by the buyer."
            ),
            "discrete_terms.user_product_preference": (
                "How well the listing photo matches the buyer's stated preference for nacho chips that still look "
                "mostly crisp under the toppings rather than fully soaked and collapsed. Use `strong_match` when "
                "chips look largely intact, `partial_match` when some sogginess shows, and `mismatch_or_uncertain` "
                "when they look soaked through or cannot be confirmed."
            ),
        },
        "continuous_bounds": {},
        "discrete_options": {
            "delivery_speed": ["rush", "standard", "batched"],
            "extra_condiments": [True, False],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 10.61,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for this all-in food delivery order before delivery speed "
                    "and condiment terms, measured in dollars. A lower price is better for you because every "
                    "dollar paid reduces your utility by 1 dollar."
                ),
                "discrete_weights.delivery_speed": (
                    "How much each delivery speed option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.extra_condiments": (
                    "How much receiving extra condiments changes your utility, measured in dollars."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated chip-texture preference changes your utility, "
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
            "c_base": 8.80,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost for fulfilling this all-in food delivery order before delivery "
                    "speed and condiment terms, measured in dollars. A higher price is better for you because "
                    "every dollar received increases your utility by 1 dollar."
                ),
                "discrete_weights.delivery_speed": (
                    "How much each delivery speed option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.extra_condiments": (
                    "How much including extra condiments changes your utility, measured in dollars."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated chip-texture preference changes your "
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
    buyer_max_price = contract_config["buyer_preferences"]["v_base"]  # Keep for backward-compatible step reward display
    seller_min_price = contract_config["seller_preferences"]["c_base"]  # Keep for backward-compatible step reward display
    
    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, name="Seller1", seller_min_price=seller_min_price)
    
    # Method 1: Create environment using registration system (recommended)
    print("Creating negotiation environment using registration system...")
    env = make(
        "Task1_basic_price_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
        environment_info={
            "platform": "DoorDash",
            "market_type": "Food Delivery",
            # "availability_status": "Available for delivery.",
            "estimated_delivery_time": "25-40 minutes",
            "restaurant_price_range": "$$",
            "restaurant_address": "1731 Westheimer Rd, Houston, TX 77098, USA",
            "delivery_distance_miles": 3.7,
            "delivery_distance_km": 6.0,
            "menu_price": 9.50,
            "delivery_fee": 2.49,
            "service_fee": 1.15,
            "quoted_total_price": 13.14,
            "pricing_rule": "Negotiated price should be treated as the all-in order total including the food item and mandatory delivery-related fees.",
            "contract_config": contract_config,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Method 2: Direct instantiation (backward compatible, but not recommended)
    # env = Task1BasicPriceNegotiation(
    #     buyer_agent=buyer,
    #     seller_agent=seller,
    #     max_rounds=20,
    #     buyer_max_price=buyer_max_price,
    #     seller_min_price=seller_min_price,
    #     environment_info={
    #         "temperature": "warm",
    #         "season": "summer",
    #         "weather": "sunny",
    #     },
    #     price_tolerance=1.0,
    # )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    user_requirement = contract_config["contrainfo"]["product_request"]
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation...")
    print("="*60)
    
    # Product image for VLM: URL from restaurantmenuchanges.csv
    product_image_url = "https://img.cdn4dd.com/cdn-cgi/image/fit=contain,width=1200,height=672,format=auto/https://doordash-static.s3.amazonaws.com/media/photosV2/47d1ce98-66be-4bc9-9f0e-c41e08b60f4a-retina-large.jpg"
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "Dripped Nachos",
            "condition": "Prepared fresh to order",
            "brand": "Dripped Birria",
            "flavor": "Nacho chips with birria beef, cheese sauce, jalapenos, onions, and cilantro",
            "size": "Single loaded nachos tray",
            "original_price": 9.50,
            "restaurant_address": "1731 Westheimer Rd, Houston, TX 77098, USA",
            "delivery_distance_miles": 3.7,
            "delivery_distance_km": 6.0,
            "delivery_fee": 2.49,
            "service_fee": 1.15,
            "quoted_total_price": 13.14,
            # "availability_status": "Available for delivery.",
            "product_category": "Food Delivery › Mexican › Nachos",
            "average_rating": 4.31,
            "total_reviews": 557,
            "seller_name": "Dripped Birria",
            "asin": "DD-HOU-DRIPPED-BIRRIA-NACHOS",
            "full_description": "Dripped Nachos from Dripped Birria include nacho chips topped with cheese sauce, birria de res meat, jalapenos, chopped onions, and cilantro. The restaurant is located at 1731 Westheimer Rd, Houston, TX 77098, USA, about 3.7 miles (6.0 km) away. Menu price is $9.50 with a $2.49 delivery fee and $1.15 service fee, making the all-in quoted total $13.14 from DoorDash menu data.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task22_s19_food_delivery_4",
        "category": "Food Delivery",
        "scenario": "Dripped Nachos order from Dripped Birria",
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
        
        # Display step rewards for each round with detailed calculation
        if 'step_seller_reward' in info or 'step_buyer_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_seller_reward' in info:
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            if 'step_buyer_reward' in info:
                if 'step_seller_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            if 'step_seller_reward' in info and info.get('seller_price') is not None:
                seller_price = info.get('seller_price', 0)
                seller_min = env.seller_min_price
                if seller_min is not None:
                    seller_profit = seller_price - seller_min
                    weighted_seller_profit = seller_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_price={seller_price:.2f}, seller_min={seller_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price={seller_price:.2f}, seller_min not specified, round={info['round']})")
            elif 'step_seller_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            if 'step_buyer_reward' in info and info.get('buyer_price') is not None:
                buyer_price = info.get('buyer_price', 0)
                buyer_max = env.buyer_max_price
                if buyer_max is not None:
                    buyer_savings = buyer_max - buyer_price
                    weighted_buyer_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer_reward']:.2f} (buyer_max={buyer_max}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price={buyer_price:.2f}, buyer_max not specified, round={info['round']})")
            elif 'step_buyer_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
        
        # If this is the final round (agreed or timeout), display score calculations after Step Rewards
        if done:
            # Print score calculations after Step Rewards
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
            print(f"Final Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Total Reward: {reward:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
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
            # current_round has been incremented to reflect the completed round
            actual_rounds = info.get('round', 0)
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "total_rounds": actual_rounds,
                "total_reward": float(reward) if reward is not None else None,
                "seller_reward": info.get('seller_reward'),
                "buyer_reward": info.get('buyer_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
                "product_info": {
                    "name": "Dripped Nachos",
                    "brand": "Dripped Birria",
                    "flavor": "Nacho chips with birria beef, cheese sauce, jalapenos, onions, and cilantro",
                    "original_price": 9.50,
                    "restaurant_address": "1731 Westheimer Rd, Houston, TX 77098, USA",
                    "delivery_distance_miles": 3.7,
                    "delivery_distance_km": 6.0,
                    "delivery_fee": 2.49,
                    "service_fee": 1.15,
                    "quoted_total_price": 13.14,
                    "product_category": "Food Delivery › Mexican › Nachos",
                    "average_rating": 4.31,
                    "total_reviews": 557,
                    "asin": "DD-HOU-DRIPPED-BIRRIA-NACHOS"
                },
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "single_buyer_product_seller"
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
        
        # Save output text (we'll create a simple output file with key information)
        output_file = run_dir / "Task22_s19_food_delivery_4_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task22 Scenario 19: Food Delivery (Dripped Nachos) Negotiation Results\n")
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
            f.write(f"  Seller Price: ${results['seller_price']:.2f}" if results.get('seller_price') else "  Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer Price: ${results['buyer_price']:.2f}" if results.get('buyer_price') else "  Buyer Price: Not specified")
            f.write("\n")
            if results.get('agreed_price'):
                f.write(f"  Agreed Price: ${results['agreed_price']:.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('seller_reward') is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
            if results.get('buyer_reward') is not None:
                f.write(f"  Buyer Reward: {results['buyer_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task22 Scenario 19: Food Delivery Negotiation (Dripped Nachos)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VLM model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'gpt-4-vision-preview'). Default: gpt-4o"
    )
    args = parser.parse_args()
    main(model_name=args.model)

