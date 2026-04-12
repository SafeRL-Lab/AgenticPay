"""Task2 Parallel Three-Buyer Three-Seller Two-Product Negotiation Example

Demonstrates how to use the Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation to negotiate with
three buyers and three sellers in parallel for two products.
Prices represent total price for both products.
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

from agenticpay.envs.multi_buyer_multi_products_multi_seller.Task2_parallel_three_buyer_three_seller_two_product_negotiation import Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation
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
try:
    from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance, OPENAI_API_KEY
except ImportError:
    # Default values if config not available
    reward_weights = {"buyer_savings": 1.0, "seller_profit": 1.0, "time_cost": 0.1}
    buyer_reward_aggregation = "average"
    seller_reward_aggregation = "average"
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


def main(model_name=None):
    """Main function: Demonstrates multi-buyer multi-seller multi-product negotiation flow
    
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
    
    # Use OpenAIVLM (Vision Language Model) - same pattern as Task1_basic_price_negotiation_api
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

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
    buyer1_max_price = 180.0  # Maximum acceptable total purchase price for buyer1 (confidential, for both products)
    buyer2_max_price = 190.0  # Maximum acceptable total purchase price for buyer2 (confidential, for both products)
    buyer3_max_price = 195.0  # Maximum acceptable total purchase price for buyer3 (confidential, for both products)
    seller1_min_price = 150.0  # Minimum acceptable total selling price for seller1 (confidential, for both products)
    seller2_min_price = 160.0  # Minimum acceptable total selling price for seller2 (confidential, for both products)
    seller3_min_price = 170.0  # Minimum acceptable total selling price for seller3 (confidential, for both products)
    
    buyer1 = BuyerAgent(model=model, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, buyer_max_price=buyer2_max_price)
    buyer3 = BuyerAgent(model=model, buyer_max_price=buyer3_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(model=model, seller_min_price=seller3_min_price)
    
    # Create environment
    print("Creating multi-buyer multi-seller multi-product negotiation environment...")
    env = Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        buyer3_agent=buyer3,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=max_rounds,
        initial_seller1_price=250.0,  # Initial total price offered by seller1 for both products
        initial_seller2_price=260.0,  # Initial total price offered by seller2 for both products
        initial_seller3_price=270.0,  # Initial total price offered by seller3 for both products
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential, for both products)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential, for both products)
        buyer3_max_price=buyer3_max_price,  # Buyer3 total max price (confidential, for both products)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential, for both products)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential, for both products)
        seller3_min_price=seller3_min_price,  # Seller3 total min price (confidential, for both products)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
        buyer_reward_aggregation=buyer_reward_aggregation,  # Buyer reward aggregation method
        seller_reward_aggregation=seller_reward_aggregation,  # Seller reward aggregation method
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices
    # The product_info should contain a list of two products
    product_info = {
        "products": [
            {
                "name": "Premium Winter Jacket",
                "brand": "Mountain Gear",
                "price": 120.0,  # Individual price of first product
                "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                "condition": "New",
                "material": "Gore-Tex",
            },
            {
                "name": "Running Shoes",
                "brand": "SportMax",
                "price": 80.0,  # Individual price of second product
                "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
                "condition": "New",
                "material": "Mesh and Synthetic",
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
    user_requirement = "I need a high-quality winter jacket and a pair of running shoes for my outdoor activities"
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation with three buyers and three sellers for two products...")
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
        "task": "Task2_parallel_three_buyer_three_seller_two_product_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round: buyers respond first, then sellers respond (seeing buyers' messages)
        # Get buyer1's responses
        buyer1_action_seller1 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s1"],
            current_state=observation
        )
        
        buyer1_action_seller2 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s2"],
            current_state=observation
        )
        
        buyer1_action_seller3 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s3"],
            current_state=observation
        )
        
        # Get buyer2's responses
        buyer2_action_seller1 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s1"],
            current_state=observation
        )
        
        buyer2_action_seller2 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s2"],
            current_state=observation
        )
        
        buyer2_action_seller3 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s3"],
            current_state=observation
        )
        
        # Get buyer3's responses
        buyer3_action_seller1 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s1"],
            current_state=observation
        )
        
        buyer3_action_seller2 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s2"],
            current_state=observation
        )
        
        buyer3_action_seller3 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s3"],
            current_state=observation
        )
        
        # Create updated conversation histories that include buyers' responses
        # So sellers can see buyers' messages before responding
        updated_conversation_history_b1s1 = observation["conversation_history_b1s1"].copy()
        updated_conversation_history_b1s2 = observation["conversation_history_b1s2"].copy()
        updated_conversation_history_b1s3 = observation["conversation_history_b1s3"].copy()
        updated_conversation_history_b2s1 = observation["conversation_history_b2s1"].copy()
        updated_conversation_history_b2s2 = observation["conversation_history_b2s2"].copy()
        updated_conversation_history_b2s3 = observation["conversation_history_b2s3"].copy()
        updated_conversation_history_b3s1 = observation["conversation_history_b3s1"].copy()
        updated_conversation_history_b3s2 = observation["conversation_history_b3s2"].copy()
        updated_conversation_history_b3s3 = observation["conversation_history_b3s3"].copy()
        
        if buyer1_action_seller1:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b1s1.append({
                "role": "buyer",
                "content": buyer1_action_seller1,
                "round": current_round
            })
        
        if buyer1_action_seller2:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b1s2.append({
                "role": "buyer",
                "content": buyer1_action_seller2,
                "round": current_round
            })
        
        if buyer1_action_seller3:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b1s3.append({
                "role": "buyer",
                "content": buyer1_action_seller3,
                "round": current_round
            })
        
        if buyer2_action_seller1:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b2s1.append({
                "role": "buyer",
                "content": buyer2_action_seller1,
                "round": current_round
            })
        
        if buyer2_action_seller2:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b2s2.append({
                "role": "buyer",
                "content": buyer2_action_seller2,
                "round": current_round
            })
        
        if buyer2_action_seller3:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b2s3.append({
                "role": "buyer",
                "content": buyer2_action_seller3,
                "round": current_round
            })
        
        if buyer3_action_seller1:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b3s1.append({
                "role": "buyer",
                "content": buyer3_action_seller1,
                "round": current_round
            })
        
        if buyer3_action_seller2:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b3s2.append({
                "role": "buyer",
                "content": buyer3_action_seller2,
                "round": current_round
            })
        
        if buyer3_action_seller3:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_b3s3.append({
                "role": "buyer",
                "content": buyer3_action_seller3,
                "round": current_round
            })
        
        # Get seller1's responses (sellers can now see buyers' messages)
        seller1_action_buyer1 = seller1.respond(
            conversation_history=updated_conversation_history_b1s1,
            current_state=observation
        )
        
        seller1_action_buyer2 = seller1.respond(
            conversation_history=updated_conversation_history_b2s1,
            current_state=observation
        )
        
        seller1_action_buyer3 = seller1.respond(
            conversation_history=updated_conversation_history_b3s1,
            current_state=observation
        )
        
        # Get seller2's responses (sellers can now see buyers' messages)
        seller2_action_buyer1 = seller2.respond(
            conversation_history=updated_conversation_history_b1s2,
            current_state=observation
        )
        
        seller2_action_buyer2 = seller2.respond(
            conversation_history=updated_conversation_history_b2s2,
            current_state=observation
        )
        
        seller2_action_buyer3 = seller2.respond(
            conversation_history=updated_conversation_history_b3s2,
            current_state=observation
        )
        
        # Get seller3's responses (sellers can now see buyers' messages)
        seller3_action_buyer1 = seller3.respond(
            conversation_history=updated_conversation_history_b1s3,
            current_state=observation
        )
        
        seller3_action_buyer2 = seller3.respond(
            conversation_history=updated_conversation_history_b2s3,
            current_state=observation
        )
        
        seller3_action_buyer3 = seller3.respond(
            conversation_history=updated_conversation_history_b3s3,
            current_state=observation
        )
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_action_seller1=buyer1_action_seller1,
            buyer1_action_seller2=buyer1_action_seller2,
            buyer1_action_seller3=buyer1_action_seller3,
            buyer2_action_seller1=buyer2_action_seller1,
            buyer2_action_seller2=buyer2_action_seller2,
            buyer2_action_seller3=buyer2_action_seller3,
            buyer3_action_seller1=buyer3_action_seller1,
            buyer3_action_seller2=buyer3_action_seller2,
            buyer3_action_seller3=buyer3_action_seller3,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller1_action_buyer3=seller1_action_buyer3,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2,
            seller2_action_buyer3=seller2_action_buyer3,
            seller3_action_buyer1=seller3_action_buyer1,
            seller3_action_buyer2=seller3_action_buyer2,
            seller3_action_buyer3=seller3_action_buyer3
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_seller3_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_buyer3_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer3: {info['step_buyer3_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            if 'step_seller3_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller3: {info['step_seller3_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b1s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer1_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer1_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b2s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer2_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer2_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer3 step reward details
            if 'step_buyer3_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b3s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer3_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer3_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer3_reward']:.2f} (buyer3_max={env.buyer3_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s1_seller_price'
                    if info.get(price_key) is not None and env.seller1_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller1_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_min={env.seller1_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s2_seller_price'
                    if info.get(price_key) is not None and env.seller2_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller2_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_min={env.seller2_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            # Seller3 step reward details
            if 'step_seller3_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s3_seller_price'
                    if info.get(price_key) is not None and env.seller3_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller3_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller3_reward']:.2f} (seller3_min={env.seller3_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
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
                final_deal_price = info.get('final_deal_price') or 0
                print(f"Final Deal Total Price: ${final_deal_price:.2f}")
            # Display all 9 buyer-seller pairs
            for buyer_id in [1, 2, 3]:
                for seller_id in [1, 2, 3]:
                    buyer_price_key = f'b{buyer_id}s{seller_id}_buyer_price'
                    seller_price_key = f'b{buyer_id}s{seller_id}_seller_price'
                    buyer_price = info.get(buyer_price_key) or 0
                    seller_price = info.get(seller_price_key) or 0
                    print(f"Buyer{buyer_id}-Seller{seller_id} Total Prices: Buyer=${buyer_price:.2f} | Seller=${seller_price:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'buyer3_reward' in info:
                print(f"Buyer3 Reward: {info['buyer3_reward']:.3f}")
            if 'seller1_reward' in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if 'seller2_reward' in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
            if 'seller3_reward' in info:
                print(f"Seller3 Reward: {info['seller3_reward']:.3f}")
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
                "b1s3_buyer_price": info.get('b1s3_buyer_price'),
                "b1s3_seller_price": info.get('b1s3_seller_price'),
                "b2s1_buyer_price": info.get('b2s1_buyer_price'),
                "b2s1_seller_price": info.get('b2s1_seller_price'),
                "b2s2_buyer_price": info.get('b2s2_buyer_price'),
                "b2s2_seller_price": info.get('b2s2_seller_price'),
                "b2s3_buyer_price": info.get('b2s3_buyer_price'),
                "b2s3_seller_price": info.get('b2s3_seller_price'),
                "b3s1_buyer_price": info.get('b3s1_buyer_price'),
                "b3s1_seller_price": info.get('b3s1_seller_price'),
                "b3s2_buyer_price": info.get('b3s2_buyer_price'),
                "b3s2_seller_price": info.get('b3s2_seller_price'),
                "b3s3_buyer_price": info.get('b3s3_buyer_price'),
                "b3s3_seller_price": info.get('b3s3_seller_price'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get('buyer1_reward'),
                "buyer2_reward": info.get('buyer2_reward'),
                "buyer3_reward": info.get('buyer3_reward'),
                "seller1_reward": info.get('seller1_reward'),
                "seller2_reward": info.get('seller2_reward'),
                "seller3_reward": info.get('seller3_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "buyer3_max_price": buyer3_max_price,
                "seller1_min_price": seller1_min_price,
                "seller2_min_price": seller2_min_price,
                "seller3_min_price": seller3_min_price,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nMulti-buyer multi-seller multi-product negotiation completed!")
    
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
        output_file = run_dir / "Task2_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task2: Parallel Three-Buyer Three-Seller Two-Product Negotiation Results\n")
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
            for buyer_id in [1, 2, 3]:
                for seller_id in [1, 2, 3]:
                    buyer_price = results.get(f'b{buyer_id}s{seller_id}_buyer_price')
                    seller_price = results.get(f'b{buyer_id}s{seller_id}_seller_price')
                    if buyer_price is not None and seller_price is not None:
                        f.write(f"  Buyer{buyer_id}-Seller{seller_id}: Buyer=${buyer_price:.2f} | Seller=${seller_price:.2f}\n")
                    else:
                        f.write(f"  Buyer{buyer_id}-Seller{seller_id}: Not specified\n")
            f.write("\n")
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
            if results.get('buyer3_reward') is not None:
                f.write(f"  Buyer3 Reward: {results['buyer3_reward']:.3f}\n")
            if results.get('seller1_reward') is not None:
                f.write(f"  Seller1 Reward: {results['seller1_reward']:.3f}\n")
            if results.get('seller2_reward') is not None:
                f.write(f"  Seller2 Reward: {results['seller2_reward']:.3f}\n")
            if results.get('seller3_reward') is not None:
                f.write(f"  Seller3 Reward: {results['seller3_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task2: Parallel Three-Buyer Three-Seller Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

