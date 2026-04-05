"""Task7 Scenario 3: Riflescope & Epson Printer - Sequential Two-Seller Per One Product Negotiation

Buyer negotiating with two sellers: Seller1 offers Crimson Trace Riflescope, Seller2 offers Epson thermal receipt printer.
Buyer chooses one seller per round to negotiate with.
Category: Sports & Outdoors / Office Electronics
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

from agenticpay.envs.multi_products_multi_seller.Task3_sequential_two_seller_per_one_product_negotiation import Task3SequentialTwoSellerPerOneProductNegotiation
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


def extract_seller_choice(buyer_response: str, observation: dict) -> int:
    """Extract seller choice from buyer's response
    
    Buyer should indicate which seller they want to negotiate with.
    Look for patterns like "seller 1", "seller1", "first seller", etc.
    
    Args:
        buyer_response: Buyer's response text
        observation: Current observation from environment
        
    Returns:
        1 or 2, indicating which seller buyer wants to negotiate with
    """
    response_lower = buyer_response.lower()
    
    # Look for explicit seller mentions
    if re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
        return 2
    elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
        return 1
    
    # If no explicit mention, try to infer from context
    # Check if buyer mentions prices or other indicators
    seller1_price = observation.get("seller1_price")
    seller2_price = observation.get("seller2_price")
    
    # If buyer mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', buyer_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller1_price is not None and abs(mentioned_price - seller1_price) < 5:
            return 1
        elif seller2_price is not None and abs(mentioned_price - seller2_price) < 5:
            return 2
    
    # Default: if no clear indication, check which seller has been negotiated with more
    # or which has a better price
    if seller1_price is not None and seller2_price is not None:
        # Choose the one with lower price if both available
        return 1 if seller1_price <= seller2_price else 2
    elif seller1_price is not None:
        return 1
    elif seller2_price is not None:
        return 2
    
    # Final default: seller1
    return 1


def main(model_name=None):
    """Main function: Demonstrates sequential multi-seller negotiation flow with different products
    
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
    
    # Use OpenAIVLM (Vision Language Model) for product negotiation with product images (图文)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview
    model = OpenAIVLM(model=model_name, api_key=api_key)
    
    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    buyer_max_price = 300.0  # Maximum acceptable price for buyer (confidential) - can afford either product
    seller1_min_price = 180.0  # Minimum acceptable price for seller1 - Crimson Trace Riflescope (confidential)
    seller2_min_price = 250.0  # Minimum acceptable price for seller2 - Epson Printer (confidential)
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=218.79,  # Initial price for Crimson Trace Riflescope (list price)
        initial_seller2_price=320.0,  # Initial price for Epson thermal receipt printer (list price)
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "comparison_enabled": True,
        },
        price_tolerance=0,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Outdoor enthusiast and small business owner. Values quality optics for hunting and reliable office equipment. Prefers to buy from reputable sellers with good reviews. Interested in either a riflescope or a thermal receipt printer."
    print(f"User Profile: {user_profile}")
    
    # Get user requirement
    # Use default requirement for automatic running
    user_requirement = "I'm looking for either a Crimson Trace Riflescope for hunting or an Epson thermal receipt printer for my small business. Prefer good value and reliable products."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment with different products for each seller
    # Product 1: Crimson Trace Riflescope (from Task6 example)
    # Product 2: Epson thermal receipt printer (from sampled_products2.jsonl, 3rd sample)
    print("\n" + "="*60)
    print("Starting new sequential negotiation with two sellers (Riflescope vs Epson Printer)...")
    print("="*60)
    
    riflescope_image_url = "https://m.media-amazon.com/images/I/31j7DdlfrOL.jpg"
    epson_image_url = "https://m.media-amazon.com/images/I/51BzGMyEVfL.jpg"
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info={
            "name": "Crimson Trace Brushline Pro Riflescope with Lightweight Solid Construction, Scope Caps and Lens Cloth for Hunting, Shooting and Outdoor",
            "condition": "New",
            "brand": "Visit the Crimson Trace Store",
            "model": "Brushline Pro Riflescope 2.5-10x42mm CT Plex Reticle",
            "style": "2.5-10x42mm Plex",
            "original_price": 218.79,
            "availability_quantity": 14,
            "availability_status": "Only 14 left in stock (more on the way).",
            "product_category": "Sports & Outdoors › Hunting & Fishing › Shooting › Optics › Gun Scopes › Rifle Scopes",
            "average_rating": 4.3,
            "total_reviews": 28,
            "seller_name": "Amazon.com",
            "asin": "B08GS6B87J",
            "full_description": "SPECS: 2.5-10 magnification with a 42mm lens diameter, aerospace grade 1\" tube and weighs 16.6 oz - FOV Range: 40.3 ft Min - 10.1 ft Max. ACCURACY: Features a second focal plane, non-illuminated, CT Plex reticle with a 4\" eye relief, 1/4\" click value and quick spring-loaded zero reset capped turrets. EASE OF USE: Windage (right side), elevation (top) knobs are capped and can be easily unscrewed and adjusted when sighting in at the range by turning with your fingers (no tool required). DURABLE: Constructed of lightweight anodized aluminum with multi-coated lenses and is waterproof, shockproof and nitrogen purged to prevent fogging. INCLUDES: Lens cloth and scope caps.",
            "image_url": riflescope_image_url,
        },
        seller2_product_info={
            "name": "Epson C31CB10023 TM-T20 Readyprint Thermal Receipt Printer, Ethernet Interface, Without Cable, Dark Grey",
            "condition": "New",
            "brand": "Visit the Epson Store",
            "model": "C31CB10023",
            "original_price": 320.0,
            "availability_status": "In stock. Usually ships within 3 to 4 days.",
            "product_category": "Office Products › Office Electronics",
            "average_rating": 4.1,
            "total_reviews": 4,
            "seller_name": "SourceLink Technologies",
            "asin": "B00A0WG5KW",
            "full_description": "For nearly 40 years, Epson has led the industry in developing innovative, reliable, high-performance products. From scanners to printers to 3D projectors, our award-winning technology brings your images to life. Epson Headquartered and established on the shore of Lake Suwa in Nagano, Japan.",
            "image_url": epson_image_url,
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task7_s3_riflescope_epson_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round, buyer chooses one seller to negotiate with
        # Buyer can see both sellers' information in the observation
        # Let buyer decide which seller to negotiate with and provide negotiation message
        # We'll use a combined conversation history that includes both sellers' conversations
        combined_history = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_seller1", []):
            combined_history.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_seller2", []):
            combined_history.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        
        # Get buyer's response - buyer should indicate which seller they want to negotiate with
        buyer_response = buyer.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "You are negotiating with two sellers: Seller 1 offers Crimson Trace Riflescope, Seller 2 offers Epson thermal receipt printer. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'."
            }
        )
        
        # Extract seller choice from buyer's response
        selected_seller = extract_seller_choice(buyer_response, observation)
        print(f"\n[Buyer chooses to negotiate with Seller {selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        # The response may include the choice statement, which is fine as it's buyer's natural expression
        buyer_action = buyer_response
        
        # Get the conversation history for the selected seller
        if selected_seller == 1:
            conversation_history = observation["conversation_history_seller1"]
        else:
            conversation_history = observation["conversation_history_seller2"]
        
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
        
        # Get the selected seller's response (seller can now see buyer's message)
        if selected_seller == 1:
            seller_action = seller1.respond(
                conversation_history=updated_conversation_history,
                current_state=observation
            )
        else:
            seller_action = seller2.respond(
                conversation_history=updated_conversation_history,
                current_state=observation
            )
        
        # Execute step with selected seller and actions
        observation, reward, terminated, truncated, info = env.step(
            selected_seller=selected_seller,
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if 'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_buyer_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer_reward' in info:
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer step reward details
            if 'step_buyer_reward' in info:
                buyer_price = None
                if info.get('current_selected_seller') == 1:
                    buyer_price = info.get('buyer_price_seller1')
                elif info.get('current_selected_seller') == 2:
                    buyer_price = info.get('buyer_price_seller2')
                
                if buyer_price is not None and env.buyer_max_price is not None:
                    buyer_savings = env.buyer_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer_reward']:.2f} (buyer_max={env.buyer_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info and info.get('seller1_price') is not None:
                seller1_price = info.get('seller1_price', 0)
                seller1_min = env.seller1_min_price
                if seller1_min is not None:
                    seller1_profit = seller1_price - seller1_min
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_price={seller1_price:.2f}, seller1_min={seller1_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price={seller1_price:.2f}, seller1_min not specified, round={info['round']})")
            elif 'step_seller1_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info and info.get('seller2_price') is not None:
                seller2_price = info.get('seller2_price', 0)
                seller2_min = env.seller2_min_price
                if seller2_min is not None:
                    seller2_profit = seller2_price - seller2_min
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={seller2_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price={seller2_price:.2f}, seller2_min not specified, round={info['round']})")
            elif 'step_seller2_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
        
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
            if info.get('selected_seller'):
                print(f"Final Selected Seller: Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
                # Display product info for selected seller
                if info['selected_seller'] == 1:
                    product_info = info.get('seller1_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
                elif info['selected_seller'] == 2:
                    product_info = info.get('seller2_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
            seller1_price = info.get('seller1_price', 0) or 0
            buyer_price_seller1 = info.get('buyer_price_seller1', 0) or 0
            seller2_price = info.get('seller2_price', 0) or 0
            buyer_price_seller2 = info.get('buyer_price_seller2', 0) or 0
            print(f"Seller1 Prices: Seller=${seller1_price:.2f} | Buyer=${buyer_price_seller1:.2f}")
            print(f"Seller2 Prices: Seller=${seller2_price:.2f} | Buyer=${buyer_price_seller2:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
            if 'seller1_reward' in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if 'seller2_reward' in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
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
            seller1_product_info = info.get('seller1_product_info', {})
            seller2_product_info = info.get('seller2_product_info', {})
            # current_round has been incremented to reflect the completed round
            actual_rounds = info.get('round', 0)
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "seller1_price": info.get('seller1_price'),
                "seller2_price": info.get('seller2_price'),
                "buyer_price_seller1": info.get('buyer_price_seller1'),
                "buyer_price_seller2": info.get('buyer_price_seller2'),
                "total_rounds": actual_rounds,
                "total_reward": float(reward) if reward is not None else None,
                "buyer_reward": info.get('buyer_reward'),
                "seller1_reward": info.get('seller1_reward'),
                "seller2_reward": info.get('seller2_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller1_min_price": seller1_min_price,
                "seller2_min_price": seller2_min_price,
                "seller1_product_info": seller1_product_info,
                "seller2_product_info": seller2_product_info,
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
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_products_multi_seller"
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
        output_file = run_dir / "Task7_s3_riflescope_epson_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task7 Scenario 3: Riflescope & Epson - Sequential Two-Seller Per One Product Negotiation Results\n")
            f.write("Category: Sports & Outdoors / Office Electronics\n")
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
            if results.get('selected_seller'):
                f.write(f"Final Selected Seller: Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n")
                selected_product = results.get('seller1_product_info', {}) if results['selected_seller'] == 1 else results.get('seller2_product_info', {})
                f.write(f"Selected Product: {selected_product.get('name', 'N/A')} by {selected_product.get('brand', 'N/A')}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
            f.write("Products:\n")
            seller1_product = results.get('seller1_product_info', {})
            price1 = seller1_product.get('original_price', seller1_product.get('price', 0)) or 0
            f.write(f"  Seller1 Product: {seller1_product.get('name', 'N/A')} by {seller1_product.get('brand', 'N/A')} (${price1:.2f})\n")
            seller2_product = results.get('seller2_product_info', {})
            price2 = seller2_product.get('original_price', seller2_product.get('price', 0)) or 0
            f.write(f"  Seller2 Product: {seller2_product.get('name', 'N/A')} by {seller2_product.get('brand', 'N/A')} (${price2:.2f})\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('buyer_reward') is not None:
                f.write(f"  Buyer Reward: {results['buyer_reward']:.3f}\n")
            if results.get('seller1_reward') is not None:
                f.write(f"  Seller1 Reward: {results['seller1_reward']:.3f}\n")
            if results.get('seller2_reward') is not None:
                f.write(f"  Seller2 Reward: {results['seller2_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task7 Scenario 3: Riflescope & Epson - Sequential Two-Seller Per One Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

