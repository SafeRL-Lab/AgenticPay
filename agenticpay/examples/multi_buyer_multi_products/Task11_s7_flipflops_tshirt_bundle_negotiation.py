"""Task11 Scenario 7: Flip Flops & Marvel T-Shirt - Sequential Two-Buyer Two-Product Negotiation

One seller negotiating with two buyers for N/C Flip Flops + Marvel Avengers T-Shirt package.
Seller chooses one buyer per round to negotiate with.
Prices represent total price for both products (Flip Flops + T-Shirt).
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

from agenticpay.envs.multi_buyer_multi_products.Task3_sequential_two_buyer_two_product_negotiation import Task3SequentialTwoBuyerTwoProductNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.openai_vlm import OpenAIVLM
import re

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
try:
    from config import reward_weights, max_rounds, price_tolerance, OPENAI_API_KEY
except ImportError:
    # Default values if config not available
    reward_weights = {"buyer_savings": 1.0, "seller_profit": 1.0, "time_cost": 0.1}
    max_rounds = 20
    price_tolerance = 1.0
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def extract_buyer_choice(seller_response: str, observation: dict) -> int:
    """Extract buyer choice from seller's response
    
    Seller should indicate which buyer they want to negotiate with.
    Look for patterns like "buyer 1", "buyer1", "first buyer", etc.
    
    Args:
        seller_response: Seller's response text
        observation: Current observation from environment
        
    Returns:
        1 or 2, indicating which buyer seller wants to negotiate with
    """
    response_lower = seller_response.lower()
    
    # Look for explicit buyer mentions
    if re.search(r'buyer\s*[12]|first\s+buyer|buyer\s*one', response_lower):
        if re.search(r'buyer\s*2|second\s+buyer|buyer\s*two', response_lower):
            return 2
        elif re.search(r'buyer\s*1|first\s+buyer|buyer\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if seller mentions prices or other indicators
    buyer1_price = observation.get("buyer1_price")
    buyer2_price = observation.get("buyer2_price")
    seller_price_buyer1 = observation.get("seller_price_buyer1")
    seller_price_buyer2 = observation.get("seller_price_buyer2")
    
    # If seller mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', seller_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller_price_buyer1 is not None and abs(mentioned_price - seller_price_buyer1) < 5:
            return 1
        elif seller_price_buyer2 is not None and abs(mentioned_price - seller_price_buyer2) < 5:
            return 2
    
    # Default: if no clear indication, check which buyer has been negotiated with more
    # or which has a better price (higher buyer price is better for seller)
    if buyer1_price is not None and buyer2_price is not None:
        # Choose the one with higher price if both available
        return 1 if buyer1_price >= buyer2_price else 2
    elif buyer1_price is not None:
        return 1
    elif buyer2_price is not None:
        return 2
    
    # Final default: buyer1
    return 1


def main(model_name=None):
    """Main function: Demonstrates sequential multi-buyer multi-product negotiation flow
    
    Args:
        model_name: Optional model name. If None, uses default model.
    """
    
    print("Initializing model...")
    
    # OpenVLM via OpenAI-compatible API (product images passed to VLM)
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY or "token-abc123"
    openvlm_base_url = os.getenv("OPENAI_URL") or os.getenv("OPENVLM_BASE_URL", "http://localhost:8000/v1")
    openvlm_model = os.getenv("OPENVLM_MODEL", "openvlm")
    
    model = OpenAIVLM(
        model=model_name or openvlm_model,
        api_key=api_key,
        base_url=openvlm_base_url,
    )
    
    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total price for both products (Flip Flops + T-Shirt)
    print("Creating agents...")
    buyer1_max_price = 40.0  # Maximum acceptable total price for buyer1 (confidential)
    buyer2_max_price = 45.0  # Maximum acceptable total price for buyer2 (confidential)
    seller_min_price = 25.0  # Minimum acceptable total price for seller (confidential)
    
    buyer1 = BuyerAgent(model=model, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, buyer_max_price=buyer2_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer multi-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_price=38.0,  # Initial total price offered by seller (Flip Flops + T-Shirt)
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential)
        seller_min_price=seller_min_price,  # Seller total min price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
        },
        price_tolerance=0,  # Set price_tolerance to 0
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Two people looking for summer footwear and casual Marvel fan apparel. Buyer1 seeks good value for flip flops and T-shirt. Buyer2 values comfort and officially licensed merchandise."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices (Product 1 from Task10_s7_sandals, Product 2 from sampled_products2.jsonl line 7)
    # Product 1: N/C Mens Flip Flops
    # Product 2: Marvel Avengers Captain America T-Shirt
    product_info = {
        "products": [
            {
                "name": "N/C Mens Flip Flops Thong Sandals Yoga Foam Slippers 44 R011 Black",
                "brand": "Brand: N/C",
                "price": 17.99,
                "condition": "New",
                "size": "44",
                "color": "R011 Black",
                "availability_status": "In stock. Usually ships within 3 to 4 days.",
                "product_category": "Clothing, Shoes & Jewelry › Men › Shoes › Sandals",
                "average_rating": 4.0,
                "total_reviews": 0,
                "seller_name": "changqia'w",
                "asin": "B0989VY7D8",
                "full_description": "Men's flip flops: the skin contact part is made of cloth so you can walk without friction or sharpness. Even if used for a long time, it will not cause blisters. Antiskid comfort with good spiral antiskid pattern at the bottom. Arch support provides good walking stability and keeps your feet comfortable. Upper material: mesh. Sole material: PVC. Waterproof and suitable for all seasons. Perfect for beach, pool, and casual wear. Sizes: 39-45.",
                "image_url": "https://m.media-amazon.com/images/I/61vR1ZJ9u3S.jpg",
            },
            {
                "name": "Marvel Avengers: Endgame Captain America America's Language T-Shirt",
                "brand": "Brand: Marvel",
                "price": 22.99,
                "condition": "New",
                "availability_status": "In Stock.",
                "product_category": "Clothing, Shoes & Jewelry › Novelty & More › Clothing › Novelty › Women › Tops & Tees › T-Shirts",
                "average_rating": 5,
                "total_reviews": 2,
                "seller_name": "",
                "asin": "B07XPR3R7N",
                "full_description": "Team up with what is left of the Avengers to fix the damage that Thanos has caused to the universe. You'll find the perfect gear within this collection of Officially Licensed Marvel Avengers: Endgame tee shirts, sweatshirts, and hoodies!",
                "small_description": ["Officially Licensed Marvel Apparel", "19MARF00431A-001", "Lightweight, Classic fit, Double-needle sleeve and bottom hem"],
                "image_url": "https://m.media-amazon.com/images/I/A13usaonutL.png",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Flip Flops + T-Shirt Package):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Package Price: ${total_product_price:.2f}")
    
    # Get user requirement
    user_requirement = "I'm looking for men's flip flops, size 44, black color with cloth upper for beach and pool, plus a Marvel Avengers Captain America T-shirt in black, men's fit. Prefer comfortable and good value."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation for Flip Flops + Marvel T-Shirt package...")
    print("Seller choosing between two buyers for Flip Flops and Marvel Avengers T-Shirt")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task11_s7_flipflops_tshirt_bundle_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round: seller chooses one buyer to negotiate with, then buyer responds first, then seller responds
        # Seller can see both buyers' information in the observation
        # Let seller decide which buyer to negotiate with
        # We'll use a combined conversation history that includes both buyers' conversations
        combined_history = []
        # Add buyer1 messages with prefix
        for msg in observation.get("conversation_history_buyer1", []):
            combined_history.append({
                **msg,
                "content": f"[Buyer 1] {msg['content']}"
            })
        # Add buyer2 messages with prefix
        for msg in observation.get("conversation_history_buyer2", []):
            combined_history.append({
                **msg,
                "content": f"[Buyer 2] {msg['content']}"
            })
        
        # Get seller's choice - seller should indicate which buyer they want to negotiate with
        seller_choice_response = seller.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "You are negotiating with two buyers for two products. Each round, you need to choose ONE buyer to negotiate with. Please clearly indicate which buyer (1 or 2) you want to negotiate with, for example: 'I want to negotiate with buyer 1' or 'Let me talk to buyer 2'. Prices represent total price for both products."
            }
        )
        
        # Extract buyer choice from seller's response
        selected_buyer = extract_buyer_choice(seller_choice_response, observation)
        print(f"\n[Seller chooses to negotiate with Buyer {selected_buyer} this round]")
        
        # Get the conversation history for the selected buyer
        if selected_buyer == 1:
            conversation_history = observation["conversation_history_buyer1"]
        else:
            conversation_history = observation["conversation_history_buyer2"]
        
        # Get the selected buyer's response first (buyer responds based on current history)
        if selected_buyer == 1:
            buyer_action = buyer1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:
            buyer_action = buyer2.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = conversation_history.copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's negotiation response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
            current_state=observation
        )
        
        # Execute step with selected buyer and actions (order: buyer -> seller)
        observation, reward, terminated, truncated, info = env.step(
            selected_buyer=selected_buyer,
            seller_action=seller_action,
            buyer_action=buyer_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer1_price = info.get('buyer1_price')
                if buyer1_price is not None and env.buyer1_max_price is not None:
                    buyer1_savings = env.buyer1_max_price - buyer1_price
                    weighted_savings = buyer1_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer1_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer1_price={buyer1_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer1_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer2_price = info.get('buyer2_price')
                if buyer2_price is not None and env.buyer2_max_price is not None:
                    buyer2_savings = env.buyer2_max_price - buyer2_price
                    weighted_savings = buyer2_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer2_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer2_price={buyer2_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer2_price not specified, round={info['round']})")
            
            # Seller step reward details
            if 'step_seller_reward' in info:
                seller_price = None
                if info.get('current_selected_buyer') == 1:
                    seller_price = info.get('seller_price_buyer1')
                elif info.get('current_selected_buyer') == 2:
                    seller_price = info.get('seller_price_buyer2')
                
                if seller_price is not None and env.seller_min_price is not None:
                    seller_profit = seller_price - env.seller_min_price
                    weighted_seller_profit = seller_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_price={seller_price:.2f}, seller_min={env.seller_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
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
            if info.get('selected_buyer'):
                print(f"Final Selected Buyer: Buyer {info['selected_buyer']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
            buyer1_price = info.get('buyer1_price', 0) or 0
            seller_price_buyer1 = info.get('seller_price_buyer1', 0) or 0
            buyer2_price = info.get('buyer2_price', 0) or 0
            seller_price_buyer2 = info.get('seller_price_buyer2', 0) or 0
            print(f"Buyer1 Total Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Total Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
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
                "selected_buyer": info.get('selected_buyer'),
                "final_deal_price": info.get('final_deal_price'),
                "buyer1_price": info.get('buyer1_price'),
                "buyer2_price": info.get('buyer2_price'),
                "seller_price_buyer1": info.get('seller_price_buyer1'),
                "seller_price_buyer2": info.get('seller_price_buyer2'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get('buyer1_reward'),
                "buyer2_reward": info.get('buyer2_reward'),
                "seller_reward": info.get('seller_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller_min_price": seller_min_price,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nFlip Flops & T-Shirt package negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products"
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
        output_file = run_dir / "Task11_s7_flipflops_tshirt_bundle_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task11 Scenario 7: Flip Flops & Marvel T-Shirt - Sequential Two-Buyer Two-Product Negotiation Results\n")
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
            if results.get('selected_buyer'):
                f.write(f"Final Selected Buyer: Buyer {results['selected_buyer']}\n")
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
            f.write("Final Prices (Total for Both Products):\n")
            f.write(f"  Buyer1: Buyer=${results['buyer1_price']:.2f} | Seller=${results['seller_price_buyer1']:.2f}" if results.get('buyer1_price') is not None and results.get('seller_price_buyer1') is not None else "  Buyer1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2: Buyer=${results['buyer2_price']:.2f} | Seller=${results['seller_price_buyer2']:.2f}" if results.get('buyer2_price') is not None and results.get('seller_price_buyer2') is not None else "  Buyer2: Not specified")
            f.write("\n\n")
            product_info = results.get('product_info', {})
            f.write("Products:\n")
            if 'products' in product_info:
                for i, p in enumerate(product_info['products'], 1):
                    f.write(f"  {i}. {p.get('name', 'N/A')} by {p.get('brand', 'N/A')} - ${p.get('price', 0):.2f}\n")
                total_price = sum(p.get('price', 0) for p in product_info.get('products', []))
                f.write(f"  Total Product Price: ${total_price:.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('buyer1_reward') is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get('buyer2_reward') is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get('seller_reward') is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task11 Scenario 7: Flip Flops & Marvel T-Shirt - Sequential Two-Buyer Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVLM model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)

