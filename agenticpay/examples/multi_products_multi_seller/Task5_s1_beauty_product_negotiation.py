"""Task5 Scenario 1: Beauty bundle — sequential two-seller negotiation (image + text)

Buyer wants the same two beauty items from either seller and negotiates TOTAL bundle price.
Both sellers list identical products; each has a different floor price and opening offer.
Category: Daily Life Consumption
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
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM

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
    OPENAI_API_KEY = None


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
    """Sequential two-seller negotiation for the same two-product bundle (total price).
    
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
    
    # Use OpenAIVLM (Vision Language Model) for beauty product negotiation with product images (image + text)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview
    model = OpenAIVLM(model=model_name, api_key=api_key)
    
    print(f"✓ Successfully initialized: {model}")
    
    # Agents: prices are TOTAL for the two-item bundle (confidential to each party)
    print("Creating agents...")
    product_request = (
        "I want Maybelline Turquoise eyeshadow plus NOU Oliban EDT—best bundle deal. "
        "I also prefer the pressed eyeshadow pan to look smooth and even, not scraped or chipped along the surface."
    )
    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total bundle price, delivery time, return policy, packaging option, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.delivery_days, "
                "discrete_terms.return_policy, discrete_terms.packaging, and discrete_terms.user_product_preference. "
                "The price is the total bundle price for both beauty products together."
            ),
        },
        "field_descriptions": {
            "price": "The total amount of money the buyer pays for the full two-item beauty bundle, measured in US dollars.",
            "continuous_terms.delivery_days": (
                "How many days the seller can take to deliver both beauty products after the deal is made."
            ),
            "discrete_terms.return_policy": (
                "The return rule for the full bundle. `30_days` means the buyer can return the bundle within 30 days; "
                "`none` means the sale is final and returns are not allowed."
            ),
            "discrete_terms.packaging": (
                "The packaging used for shipment. `protective` means extra protection for the eyeshadow and glass EDT bottle; "
                "`standard` means normal packaging."
            ),
            "discrete_terms.user_product_preference": (
                "How well the listed units match the buyer's stated preference for a smooth, intact pressed eyeshadow surface "
                "without obvious scraping or chips. Use `strong_match` when clearly satisfied, `partial_match` when only partly satisfied, "
                "and `mismatch_or_uncertain` when not satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {
            "delivery_days": {"min": 1, "max": 7}
        },
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "packaging": ["protective", "standard"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 23.35,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for this two-item beauty bundle before delivery, return, and packaging terms, measured in dollars. "
                    "A lower total bundle price is better for you because every dollar paid reduces your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility, measured in dollars per day. "
                    "A negative number means slower delivery is worse for you."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated product preference changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
            },
            "continuous_weights": {"delivery_days": -0.35},
            "discrete_weights": {
                "return_policy": {"30_days": 1.6, "none": -1.8},
                "packaging": {"protective": 1.4, "standard": -0.4},
                "user_product_preference": {
                    "strong_match": 0.28,
                    "partial_match": 0.11,
                    "mismatch_or_uncertain": -0.22,
                },
            },
        },
    }
    seller1_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": 20.95,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost for fulfilling this two-item beauty bundle before delivery, return, and packaging terms, measured in dollars. "
                    "A higher total bundle price is better for you because every dollar received increases your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility, measured in dollars per day. "
                    "A positive number means more delivery flexibility is better for you."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated product preference changes your utility, "
                    "measured in dollars. Stronger commitments carry a small nonzero risk or handling cost."
                ),
            },
            "continuous_weights": {"delivery_days": 0.25},
            "discrete_weights": {
                "return_policy": {"30_days": -2.0, "none": 1.2},
                "packaging": {"protective": -1.0, "standard": 0.35},
                "user_product_preference": {
                    "strong_match": -0.07,
                    "partial_match": -0.035,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    seller2_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": 18.57,
            "weight_descriptions": seller1_contract_config["seller_preferences"]["weight_descriptions"],
            "continuous_weights": {"delivery_days": 0.35},
            "discrete_weights": {
                "return_policy": {"30_days": -1.6, "none": 0.9},
                "packaging": {"protective": -1.25, "standard": 0.45},
                "user_product_preference": {
                    "strong_match": -0.07,
                    "partial_match": -0.035,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    seller_contract_configs = {
        1: seller1_contract_config,
        2: seller2_contract_config,
    }
    buyer_max_price = shared_contract_fields["buyer_preferences"]["v_base"]
    seller1_min_price = seller1_contract_config["seller_preferences"]["c_base"]
    seller2_min_price = seller2_contract_config["seller_preferences"]["c_base"]
    
    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=29.45,
        initial_seller2_price=28.9,
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "comparison_enabled": True,
            "seller_contract_configs": seller_contract_configs,
        },
        price_tolerance=0,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    product1_image_url = "https://m.media-amazon.com/images/I/41IiEBGouZL.jpg"
    product2_image_url = "https://m.media-amazon.com/images/I/51gDhcURgKL.jpg"
    
    bundle_product_info = {
        "products": [
            {
                "name": "Maybelline New York Expert Wear Eyeshadow Singles, 130s Turquoise Glass Perfect Pastels, 0.09 Ounce",
                "condition": "New",
                "brand": "Maybelline New York",
                "shade": "130s Turquoise Glass Perfect Pastels",
                "size": "0.09 Ounce",
                "price": 7.5,
                "original_price": 7.98,
                "product_category": "Beauty & Personal Care › Makeup › Eyes › Eyeshadow",
                "average_rating": 4.2,
                "total_reviews": 54,
                "asin": "B0046VILG4",
                "full_description": "Easy to use. Lots to choose. All-day crease-proof wear. Rich, velvety textures. Glides on effortlessly with superior smoothness.",
                "image_url": product1_image_url,
                "listing_age": "3 days",
            },
            {
                "name": "Oriental Eau de Toilette – Natural Eau de Toilette for Men Woody Eau de Toilette Infused with Essential Oils NOU Oliban Eau de Toilette for Men – 1.7 Fl Oz",
                "condition": "New",
                "brand": "NOU",
                "price": 21.95,
                "original_price": 21.95,
                "product_category": "Beauty & Personal Care › Fragrance",
                "average_rating": 4,
                "total_reviews": 6,
                "asin": "B08XQWJX8P",
                "full_description": "NOU Oliban oriental woody EDT for men, crafted with essential oils; elemi, olibanum, patchouli, sandalwood, leather, vanilla notes. 50ml glass flacon.",
                "image_url": product2_image_url,
                "listing_age": "5 days",
            },
        ]
    }
    
    print("\n" + "="*60)
    print("Sequential negotiation: two sellers, same 2-item beauty bundle, different bundle offers...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info=bundle_product_info,
        seller2_product_info=bundle_product_info,
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task5_s1_beauty_product_negotiation",
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
                "instruction": "Two sellers offer the SAME two beauty products as one bundle. Each round pick ONE seller (use <selected_seller>) and negotiate the TOTAL bundle price for both items together."
            }
        )
        
        selected_seller = Task3SequentialTwoSellerPerOneProductNegotiation.resolve_selected_seller(
            buyer_response, observation, buyer.last_selected_seller
        )
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
                bundle = info.get('seller1_product_info', {}) or {}
                plist = bundle.get('products') or []
                if len(plist) >= 2:
                    print(f"Bundle: (1) {plist[0].get('name', 'N/A')} | (2) {plist[1].get('name', 'N/A')}")
                elif plist:
                    print(f"Selected bundle item: {plist[0].get('name', 'N/A')}")
                agreed_contract = info.get(f"agreed_contract_seller{info['selected_seller']}")
                if agreed_contract is not None:
                    print(f"Final Contract: {agreed_contract}")
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
                "agreed_contract_seller1": info.get('agreed_contract_seller1'),
                "agreed_contract_seller2": info.get('agreed_contract_seller2'),
                "buyer_utility_seller1": info.get('buyer_utility_seller1'),
                "seller_utility_seller1": info.get('seller_utility_seller1'),
                "z_max_seller1": info.get('z_max_seller1'),
                "buyer_utility_seller2": info.get('buyer_utility_seller2'),
                "seller_utility_seller2": info.get('seller_utility_seller2'),
                "z_max_seller2": info.get('z_max_seller2'),
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
                "seller_contract_configs": seller_contract_configs,
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
        output_file = run_dir / "Task5_s1_beauty_product_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task5 Scenario 1: Beauty bundle — sequential two-seller negotiation results\n")
            f.write("Category: Daily Life Consumption\n")
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
                binfo = results.get('seller1_product_info', {}) or {}
                pl = binfo.get('products') or []
                if len(pl) >= 2:
                    f.write(f"Bundle: {pl[0].get('name', 'N/A')} + {pl[1].get('name', 'N/A')}\n\n")
                agreed_contract = results.get(f"agreed_contract_seller{results['selected_seller']}")
                if agreed_contract is not None:
                    f.write(f"Final Contract: {agreed_contract}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
            f.write("Products (same bundle for both sellers):\n")
            shared = results.get('seller1_product_info', {}) or {}
            for i, p in enumerate(shared.get('products') or [], 1):
                pr = p.get('price', p.get('original_price', 0))
                f.write(f"  {i}. {p.get('name', 'N/A')} (${float(pr):.2f})\n")
            f.write("\n")
            f.write("Contract Utilities:\n")
            if results.get('z_max_seller1') is not None:
                f.write(f"  Seller1 Z_max: {results['z_max_seller1']:.3f}\n")
            if results.get('buyer_utility_seller1') is not None:
                f.write(f"  Seller1 Buyer Utility: {results['buyer_utility_seller1']:.3f}\n")
            if results.get('seller_utility_seller1') is not None:
                f.write(f"  Seller1 Seller Utility: {results['seller_utility_seller1']:.3f}\n")
            if results.get('z_max_seller2') is not None:
                f.write(f"  Seller2 Z_max: {results['z_max_seller2']:.3f}\n")
            if results.get('buyer_utility_seller2') is not None:
                f.write(f"  Seller2 Buyer Utility: {results['buyer_utility_seller2']:.3f}\n")
            if results.get('seller_utility_seller2') is not None:
                f.write(f"  Seller2 Seller Utility: {results['seller_utility_seller2']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task5 Scenario 1: Beauty bundle — sequential two-seller negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

