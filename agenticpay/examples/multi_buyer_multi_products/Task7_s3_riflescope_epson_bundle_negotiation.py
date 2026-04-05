"""Task7 Scenario 3: Riflescope & Epson Printer Bundle - Sequential Two-Buyer Two-Product Negotiation

One seller negotiating with two buyers for product bundle (Crimson Trace Riflescope + Epson thermal receipt printer).
Seller chooses one buyer per round to negotiate with.
Prices represent total price for the bundle.
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
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Use OpenAIVLM (Vision Language Model) for Riflescope & Epson bundle with product images (图文)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini
    model = OpenAIVLM(model=model_name, api_key=api_key)
    
    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total for Riflescope ($218.79) + Epson Printer ($320) = $538.79 bundle
    print("Creating agents...")
    buyer1_max_price = 530.0  # Maximum acceptable total for buyer1 (confidential, budget buyer)
    buyer2_max_price = 560.0  # Maximum acceptable total for buyer2 (confidential, higher budget)
    seller_min_price = 450.0  # Minimum acceptable total for seller (confidential)
    
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
        initial_seller_price=520.0,  # Initial total price offered by seller for Riflescope+Epson bundle
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential, for rental bundle)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential, for rental bundle)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for rental bundle)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "comparison_enabled": True,
        },
        price_tolerance=0,  # Set price_tolerance to 0
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Two buyers competing for product bundle. Buyer1 is budget buyer. Buyer2 has higher budget for quality optics and office equipment. Both value quality and reliable products."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices (from Task6 example + sampled_products2.jsonl 3rd sample)
    # Product 1: Crimson Trace Riflescope (from Task6_s3_riflescope_negotiation.py)
    # Product 2: Epson TM-T20 Receipt Printer (from sampled_products2.jsonl line 4)
    product_info = {
        "products": [
            {
                "name": "Crimson Trace Brushline Pro Riflescope with Lightweight Solid Construction, Scope Caps and Lens Cloth for Hunting, Shooting and Outdoor",
                "brand": "Visit the Crimson Trace Store",
                "price": 218.79,
                "condition": "New",
                "model": "Brushline Pro Riflescope 2.5-10x42mm CT Plex Reticle",
                "style": "2.5-10x42mm Plex",
                "availability_status": "Only 14 left in stock (more on the way).",
                "product_category": "Sports & Outdoors › Hunting & Fishing › Shooting › Optics › Gun Scopes › Rifle Scopes",
                "average_rating": 4.3,
                "total_reviews": 28,
                "seller_name": "Amazon.com",
                "asin": "B08GS6B87J",
                "full_description": "SPECS: 2.5-10 magnification with a 42mm lens diameter, aerospace grade 1\" tube and weighs 16.6 oz - FOV Range: 40.3 ft Min - 10.1 ft Max. ACCURACY: Features a second focal plane, non-illuminated, CT Plex reticle with a 4\" eye relief, 1/4\" click value and quick spring-loaded zero reset capped turrets. EASE OF USE: Windage (right side), elevation (top) knobs are capped and can be easily unscrewed and adjusted when sighting in at the range by turning with your fingers (no tool required). DURABLE: Constructed of lightweight anodized aluminum with multi-coated lenses and is waterproof, shockproof and nitrogen purged to prevent fogging. INCLUDES: Lens cloth and scope caps.",
                "image_url": "https://m.media-amazon.com/images/I/31j7DdlfrOL.jpg",
            },
            {
                "name": "Epson C31CB10023 TM-T20 Readyprint Thermal Receipt Printer, Ethernet Interface, Without Cable, Dark Grey",
                "brand": "Visit the Epson Store",
                "price": 320.0,
                "condition": "New",
                "model": "C31CB10023",
                "availability_status": "In stock. Usually ships within 3 to 4 days.",
                "product_category": "Office Products › Office Electronics",
                "average_rating": 4.1,
                "total_reviews": 4,
                "seller_name": "SourceLink Technologies",
                "asin": "B00A0WG5KW",
                "full_description": "For nearly 40 years, Epson has led the industry in developing innovative, reliable, high-performance products. From scanners to printers to 3D projectors, our award-winning technology brings your images to life. Epson Headquartered and established on the shore of Lake Suwa in Nagano, Japan.",
                "image_url": "https://m.media-amazon.com/images/I/51BzGMyEVfL.jpg",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Riflescope & Epson Bundle):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Bundle Price: ${total_product_price:.2f}")
    
    # Get user requirement
    user_requirement = "I need a Crimson Trace Riflescope for hunting and an Epson thermal receipt printer for my small business. Looking for good value, reliable products with good reviews."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation for Riflescope & Epson printer bundle...")
    print("Seller choosing between two buyers for product bundle")
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
        "task": "Task7_s3_riflescope_epson_bundle_negotiation",
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
    print("\nRiflescope & Epson printer bundle negotiation completed!")
    
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
        output_file = run_dir / "Task7_s3_riflescope_epson_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task7 Scenario 3: Riflescope & Epson Printer Bundle - Sequential Two-Buyer Two-Product Negotiation Results\n")
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
    parser = argparse.ArgumentParser(description="Task7 Scenario 3: Riflescope & Epson Printer Bundle - Sequential Two-Buyer Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

