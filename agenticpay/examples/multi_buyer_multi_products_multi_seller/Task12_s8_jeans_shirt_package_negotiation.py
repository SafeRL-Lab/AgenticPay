"""Task12 Scenario 8: Jeans & Men's Shirt Package - Sequential Two-Buyer Two-Seller Two-Product Negotiation

Two buyers negotiating with two sellers for clothing package (myhehthw Jeans + JSPOYOU Men's Shirt).
Each buyer chooses one seller per round to negotiate with.
Prices represent total price for both products.
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

from agenticpay.envs.multi_buyer_multi_products_multi_seller.Task3_sequential_two_buyer_two_seller_two_product_negotiation import Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation
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
    price_tolerance = 0
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


def extract_seller_choice(buyer_response: str, observation: dict, buyer_id: int) -> int:
    """Extract seller choice from buyer's response
    
    Buyer should indicate which seller they want to negotiate with.
    Look for patterns like "seller 1", "seller1", "first seller", etc.
    
    Args:
        buyer_response: Buyer's response text
        observation: Current observation from environment
        buyer_id: Buyer ID (1 or 2)
        
    Returns:
        1 or 2, indicating which seller buyer wants to negotiate with
    """
    response_lower = buyer_response.lower()
    
    # Look for explicit seller mentions
    if re.search(r'seller\s*[12]|first\s+seller|seller\s*one', response_lower):
        if re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
            return 2
        elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if buyer mentions prices or other indicators
    # Get prices for this buyer
    if buyer_id == 1:
        seller1_price = observation.get("b1s1_seller_price")
        seller2_price = observation.get("b1s2_seller_price")
    else:  # buyer_id == 2
        seller1_price = observation.get("b2s1_seller_price")
        seller2_price = observation.get("b2s2_seller_price")
    
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
    """Main function: Demonstrates sequential multi-buyer multi-seller multi-product negotiation flow
    
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
    # buyer_max_price and seller_min_price represent total price for both products (Jeans + Shirt)
    print("Creating agents...")
    buyer1_max_price = 35.0  # Maximum acceptable total price for buyer1 (confidential, Jeans + Shirt package)
    buyer2_max_price = 38.0  # Maximum acceptable total price for buyer2 (confidential)
    seller1_min_price = 26.0  # Minimum acceptable total selling price for seller1 (confidential)
    seller2_min_price = 24.0  # Minimum acceptable total selling price for seller2 (confidential)
    
    buyer1 = BuyerAgent(model=model, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer multi-seller multi-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=32.99,  # Initial total price offered by seller1 (Jeans + Shirt package)
        initial_seller2_price=31.99,  # Initial total price offered by seller2 (competitive pricing)
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential, for materials package)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential, for materials package)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential, for materials package)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential, for materials package)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "availability_status": "In Stock",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Two fashion-conscious shoppers looking for comfortable daily wear. Buyer1 seeks good value for jeans and casual shirt. Buyer2 values quality and fit. Both prefer stretchy denim and soft cotton. Use for casual, work, and weekend wear."
    print(f"User Profile: {user_profile}")
    
    # Define two products (Product 1 from Task11 example, Product 2 from sampled_products2.jsonl sample 8)
    # Product 1: myhehthw Women's High Waisted Jeans
    # Product 2: JSPOYOU Mens Short Sleeve Crewneck Tunic Shirt
    product_info = {
        "products": [
            {
                "name": "myhehthw Women's High Waisted Jeans for Women Distressed Ripped Jeans Slim Fit Butt Lifting Skinny Stretch Jeans Trousers",
                "brand": "myhehthw",
                "price": 22.99,
                "condition": "New",
                "material": "75% Cotton, 20% Polyester, 5% Spandex",
                "sizes": ["Small", "Medium", "Large", "X-Large", "XX-Large"],
                "availability_status": "In Stock",
                "product_category": "Clothing, Shoes & Jewelry › Women › Clothing › Jeans",
                "asin": "B09PBNZLNT",
                "full_description": "Made of soft and stretchy denim cotton fabric, breathable to wear. Comfortable and versatile for casual, school, office, shopping. Button closure. Machine washable.",
                "image_url": "https://m.media-amazon.com/images/I/31Qo5LzPPjL.jpg",
            },
            {
                "name": "JSPOYOU Mens Short Sleeve Crewneck 3D Graphic Tunic Shirts Big & Tall Tie Dye Summer Top Basic Designed Classic Cotton Shirt",
                "brand": "Brand: JSPOYOU",
                "price": 10.99,
                "condition": "New",
                "material": "Stretch 60% cotton 40%, made of superior quality modal fiber, ultra light-weight, soft & smooth",
                "sizes": ["Medium", "Large", "X-Large", "XX-Large", "3X-Large"],
                "availability_status": "In Stock",
                "product_category": "Clothing, Shoes & Jewelry › Men › Clothing › Shirts › Henleys",
                "asin": "B09NNMV9LN",
                "full_description": "Gender: Men Material: Polyester, Fleece, Faux Leather Pattern Type: Printed Style: Casual Fashion, Athletics Fit Type: Standard Closure Type: Elastic What you get: 1 PC Man T-shirt tops. Stretch 60% cotton 40%. Short/long sleeve button down tshirts, big and tall tees, graphic print tops. Great for beach, work, Halloween, Christmas, loungewear, casual look, daily, dating, school, party, holiday, gym workout, running, training, jogging.",
                "small_description": ["Please check the size chart before purchase. If you want the mens shirts to be loose, choose 1-2 size up.", "Stretch 60% cotton 40%. Made of superior quality modal fiber, ultra light-weight, soft & smooth, feels really comfortable.", "Short/long sleeve button down tshirts, big and tall tees, graphic print tops, henley shirts, hawaiian shirts, tank tops, beach shirts."],
                "image_url": "https://m.media-amazon.com/images/I/41p9dAvjOoL.jpg",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Clothing Package):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Package Price: ${total_product_price:.2f}")
    
    # Get user requirement (should describe purchasing the clothing package)
    # Use default requirement for automatic running
    user_requirement = "I'm looking for either myhehthw Women's High Waisted Jeans in black or dark wash, size Medium/Large, or JSPOYOU Mens Short Sleeve Crewneck Tunic Shirt in tie dye style. Prefer good quality and fair price for the package."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation for jeans & shirt package...")
    print("Two buyers competing with two sellers for Jeans + JSPOYOU Shirt")
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
        "task": "Task12_s8_jeans_shirt_package_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round, each buyer chooses one seller to negotiate with
        # Let buyers decide which seller to negotiate with and provide negotiation message
        
        # Build combined conversation history for buyer1 (includes both sellers' conversations)
        combined_history_b1 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b1s1", []):
            combined_history_b1.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_b1s2", []):
            combined_history_b1.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        
        # Build combined conversation history for buyer2 (includes both sellers' conversations)
        combined_history_b2 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b2s1", []):
            combined_history_b2.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_b2s2", []):
            combined_history_b2.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        
        # Get buyer1's response - buyer should indicate which seller they want to negotiate with
        buyer1_response = buyer1.respond(
            conversation_history=combined_history_b1,
            current_state={
                **observation,
                "instruction": "You are negotiating with two sellers for two products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'. Prices represent total price for both products."
            }
        )
        
        # Get buyer2's response - buyer should indicate which seller they want to negotiate with
        buyer2_response = buyer2.respond(
            conversation_history=combined_history_b2,
            current_state={
                **observation,
                "instruction": "You are negotiating with two sellers for two products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'. Prices represent total price for both products."
            }
        )
        
        # Extract seller choice from each buyer's response
        buyer1_selected_seller = extract_seller_choice(buyer1_response, observation, buyer_id=1)
        buyer2_selected_seller = extract_seller_choice(buyer2_response, observation, buyer_id=2)
        
        print(f"\n[Buyer 1 chooses to negotiate with Seller {buyer1_selected_seller} this round]")
        print(f"[Buyer 2 chooses to negotiate with Seller {buyer2_selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        buyer1_action = buyer1_response
        buyer2_action = buyer2_response
        
        # Get the conversation history for each buyer-seller pair
        # Create updated conversation histories that include buyers' responses
        # So sellers can see buyers' messages before responding
        if buyer1_selected_seller == 1:
            conversation_history_b1s1 = observation["conversation_history_b1s1"].copy()
            if buyer1_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b1s1.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round
                })
        else:
            conversation_history_b1s2 = observation["conversation_history_b1s2"].copy()
            if buyer1_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b1s2.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round
                })
        
        if buyer2_selected_seller == 1:
            conversation_history_b2s1 = observation["conversation_history_b2s1"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s1.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round
                })
        else:
            conversation_history_b2s2 = observation["conversation_history_b2s2"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s2.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round
                })
        
        # Get the selected sellers' responses (sellers can now see buyers' messages)
        seller1_action_buyer1 = None
        seller1_action_buyer2 = None
        seller2_action_buyer1 = None
        seller2_action_buyer2 = None
        
        if buyer1_selected_seller == 1:
            seller1_action_buyer1 = seller1.respond(
                conversation_history=conversation_history_b1s1,
                current_state=observation
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_history_b1s2,
                current_state=observation
            )
        
        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_history_b2s1,
                current_state=observation
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_history_b2s2,
                current_state=observation
            )
        
        # Execute step with selected sellers and actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_selected_seller=buyer1_selected_seller,
            buyer2_selected_seller=buyer2_selected_seller,
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer_price = None
                if info.get('buyer1_selected_seller') == 1:
                    buyer_price = info.get('b1s1_buyer_price')
                elif info.get('buyer1_selected_seller') == 2:
                    buyer_price = info.get('b1s2_buyer_price')
                
                if buyer_price is not None and env.buyer1_max_price is not None:
                    buyer_savings = env.buyer1_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer_price = None
                if info.get('buyer2_selected_seller') == 1:
                    buyer_price = info.get('b2s1_buyer_price')
                elif info.get('buyer2_selected_seller') == 2:
                    buyer_price = info.get('b2s2_buyer_price')
                
                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller1_price = None
                # Get price from whichever buyer selected seller1
                if info.get('buyer1_selected_seller') == 1 and info.get('b1s1_seller_price') is not None:
                    seller1_price = info.get('b1s1_seller_price')
                elif info.get('buyer2_selected_seller') == 1 and info.get('b2s1_seller_price') is not None:
                    seller1_price = info.get('b2s1_seller_price')
                # If both selected seller1, prefer higher price
                if (info.get('buyer1_selected_seller') == 1 and info.get('buyer2_selected_seller') == 1 and
                    info.get('b1s1_seller_price') is not None and info.get('b2s1_seller_price') is not None):
                    seller1_price = max(info.get('b1s1_seller_price'), info.get('b2s1_seller_price'))
                
                if seller1_price is not None and env.seller1_min_price is not None:
                    seller1_profit = seller1_price - env.seller1_min_price
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_total_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info:
                seller2_price = None
                # Get price from whichever buyer selected seller2
                if info.get('buyer1_selected_seller') == 2 and info.get('b1s2_seller_price') is not None:
                    seller2_price = info.get('b1s2_seller_price')
                elif info.get('buyer2_selected_seller') == 2 and info.get('b2s2_seller_price') is not None:
                    seller2_price = info.get('b2s2_seller_price')
                # If both selected seller2, prefer higher price
                if (info.get('buyer1_selected_seller') == 2 and info.get('buyer2_selected_seller') == 2 and
                    info.get('b1s2_seller_price') is not None and info.get('b2s2_seller_price') is not None):
                    seller2_price = max(info.get('b1s2_seller_price'), info.get('b2s2_seller_price'))
                
                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_total_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
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
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
            print(f"Buyer1-Seller1 Total Prices: Buyer=${info.get('b1s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer1-Seller2 Total Prices: Buyer=${info.get('b1s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s2_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller1 Total Prices: Buyer=${info.get('b2s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller2 Total Prices: Buyer=${info.get('b2s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s2_seller_price', 0) or 0:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
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
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_buyer": info.get('selected_buyer'),
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "b1s1_buyer_price": info.get('b1s1_buyer_price'),
                "b1s1_seller_price": info.get('b1s1_seller_price'),
                "b1s2_buyer_price": info.get('b1s2_buyer_price'),
                "b1s2_seller_price": info.get('b1s2_seller_price'),
                "b2s1_buyer_price": info.get('b2s1_buyer_price'),
                "b2s1_seller_price": info.get('b2s1_seller_price'),
                "b2s2_buyer_price": info.get('b2s2_buyer_price'),
                "b2s2_seller_price": info.get('b2s2_seller_price'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get('buyer1_reward'),
                "buyer2_reward": info.get('buyer2_reward'),
                "seller1_reward": info.get('seller1_reward'),
                "seller2_reward": info.get('seller2_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller1_min_price": seller1_min_price,
                "seller2_min_price": seller2_min_price,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nJeans & shirt package negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products_multi_seller"
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
        output_file = run_dir / "Task12_s8_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task12 Scenario 8: Jeans & Men's Shirt Package - Sequential Two-Buyer Two-Seller Two-Product Negotiation Results\n")
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
            if results.get('selected_buyer') and results.get('selected_seller'):
                f.write(f"Selected Deal: Buyer {results['selected_buyer']} - Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
            f.write("Final Prices (Total for Both Products):\n")
            f.write(f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}" if results.get('b1s1_buyer_price') is not None and results.get('b1s1_seller_price') is not None else "  Buyer1-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}" if results.get('b1s2_buyer_price') is not None and results.get('b1s2_seller_price') is not None else "  Buyer1-Seller2: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}" if results.get('b2s1_buyer_price') is not None and results.get('b2s1_seller_price') is not None else "  Buyer2-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}" if results.get('b2s2_buyer_price') is not None and results.get('b2s2_seller_price') is not None else "  Buyer2-Seller2: Not specified")
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
    parser = argparse.ArgumentParser(description="Task12 Scenario 8: Jeans & Men's Shirt Package - Sequential Two-Buyer Two-Seller Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVLM model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)
