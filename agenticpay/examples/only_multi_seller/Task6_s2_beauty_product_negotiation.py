"""Task6 Scenario 2: Beauty Product - Sequential Two-Seller Negotiation

Two sellers offering NOU Oriental Eau de Toilette (men's fragrance) on Amazon.
Buyer compares offers and chooses which seller to negotiate with each round.
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

from agenticpay.envs.only_multi_seller.Task3_sequential_two_seller_negotiation import Task3SequentialTwoSellerNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM
from agenticpay.examples.config import reward_weights, max_rounds, price_tolerance
import re


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
    if re.search(r'seller\s*[12]|first\s+seller|seller\s*one', response_lower):
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
    """Main function: Demonstrates sequential multi-seller negotiation flow
    
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
    
    # Use OpenAIVLM (Vision Language Model) for beauty product negotiation with product images (图文)
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
    print("Creating agents...")
    # NOU Oriental Eau de Toilette: pricing $21.95
    buyer_max_price = 22.0  # Maximum acceptable purchase price for buyer (confidential)
    seller1_min_price = 14.0  # Minimum acceptable selling price for seller1 (confidential)
    seller2_min_price = 15.0  # Minimum acceptable selling price for seller2 (confidential, slightly higher than seller1)
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=21.95,  # Initial price offered by seller1 (product list price)
        initial_seller2_price=22.95,  # Initial price offered by seller2 (slightly higher)
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "seller1_name": "AmTm Cosmetics",
            "seller2_name": "Fragrance World",  # Both sell same NOU fragrance
            "availability_status": "In Stock",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "Style-conscious buyer interested in men's fragrances. Values natural ingredients and essential oils. Prefers woody, oriental scents with good reviews."
    print(f"User Profile: {user_profile}")
    
    # Get user requirement
    # print("\n" + "="*60)
    # print("Please enter the product requirement you want to purchase:")
    # user_requirement = input("> ").strip()
    # if not user_requirement:
    #     print("No requirement entered, using default requirement...")
    #     user_requirement = "I need a high-quality winter jacket for cold weather"
    #     print(f"Using default requirement: {user_requirement}")
    # Use default requirement for automatic running
    user_requirement = "I'm looking for a men's oriental woody fragrance with essential oils, preferably natural ingredients, good for daily wear. Budget around $22."
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation with two sellers...")
    print("="*60)
    
    # Product from sampled_products2.jsonl - NOU Oriental Eau de Toilette (men's fragrance)
    product_image_url = "https://m.media-amazon.com/images/I/51gDhcURgKL.jpg"
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "Oriental Eau de Toilette – Natural Eau de Toilette for Men Woody Eau de Toilette Infused with Essential Oils Fragrance for Men with Oriental Woody Tones NOU Oliban Eau de Toilette for Men – 1.7 Fl Oz",
            "condition": "New",
            "brand": "Brand: nou",
            "original_price": 21.95,
            "availability_status": "In Stock.",
            "product_category": "Beauty & Personal Care › Fragrance",
            "average_rating": 4,
            "total_reviews": 6,
            "seller_name": "AmTm Cosmetics",
            "asin": "B08XQWJX8P",
            "full_description": "NOU OLIBAN ORIENTAL SCENT FOR MEN – this fragrance for men has been perfectly blended and infused with essential oils, resulting in an intriguing, mysterious scent that is powerful and appealing to women. ORIENTAL EAU DE TOILETTE CRAFTED BY FRENCH PERFUMERS – expertly crafted by French perfumers, this natural fragrance for men is created with pure and natural ingredients and infused with aromatic essential oils. OLIBAN WOODY FRAGRANCE NOTES – this natural Eau de Toilette for men features elemi and olibanum, toned down by the relaxing scent of chamomile. The heart notes are patchouli and cistus, with raised notes of sandalwood, leather, vanilla and benzoin resulting in a masculine, powerful fragrance for men.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task6_s2_beauty_product_negotiation",
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
                "instruction": "You are negotiating with two sellers. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'."
            }
        )
        
        # Extract seller choice from buyer's response
        selected_seller = extract_seller_choice(buyer_response, observation)
        print(f"\n[Buyer chooses to negotiate with Seller {selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        # The response may include the choice statement, which is fine as it's buyer's natural expression
        buyer_action = buyer_response
        
        # Get the conversation history for the selected seller
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        if selected_seller == 1:
            conversation_history = observation["conversation_history_seller1"].copy()
        else:
            conversation_history = observation["conversation_history_seller2"].copy()
        
        # Add buyer's message to the conversation history
        if buyer_action:
            current_round = observation.get("current_round", 0)
            conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get the selected seller's response (seller can now see buyer's message)
        if selected_seller == 1:
            seller_action = seller1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:
            seller_action = seller2.respond(
                conversation_history=conversation_history,
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
        if 'step_buyer_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info:
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
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "seller1_price": info.get('seller1_price'),
                "seller2_price": info.get('seller2_price'),
                "buyer_price_seller1": info.get('buyer_price_seller1'),
                "buyer_price_seller2": info.get('buyer_price_seller2'),
                "total_rounds": info.get('round', 0),
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
                "product_info": {
                    "name": "Oriental Eau de Toilette – Natural Eau de Toilette for Men Woody Eau de Toilette Infused with Essential Oils Fragrance for Men with Oriental Woody Tones NOU Oliban Eau de Toilette for Men – 1.7 Fl Oz",
                    "condition": "New",
                    "brand": "Brand: nou",
                    "original_price": 21.95,
                    "product_category": "Beauty & Personal Care › Fragrance",
                    "average_rating": 4,
                    "total_reviews": 6,
                    "asin": "B08XQWJX8P",
                    "image_url": "https://m.media-amazon.com/images/I/51gDhcURgKL.jpg",
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
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_seller"
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
        output_file = run_dir / "Task6_s2_beauty_product_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task6 Scenario 2: Beauty Product - Sequential Two-Seller Negotiation Results\n")
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
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
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
    parser = argparse.ArgumentParser(description="Task6 Scenario 2: Beauty Product - Sequential Two-Seller Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
