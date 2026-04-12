"""Task1 Multi-Product Negotiation Example

Demonstrates how to use the Task1MultiProductNegotiation to negotiate multiple products
while preserving conversation context across different products.
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
# from agenticpay.agents.product_selector_agent import ProductSelectorAgent  # Removed
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
    """Main function: Demonstrates multi-product negotiation flow
    
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
    
    # Use OpenAIVLM (Vision Language Model) - same pattern as Task1_basic_price_negotiation_api
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

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
    buyer_max_price = 100.0  # Maximum acceptable purchase price for buyer (confidential)
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    # product_selector = ProductSelectorAgent(model=llm)  # Removed
    
    # Create environment using registration system
    print("Creating multi-product negotiation environment...")
    env = make(
        "Task1_multi_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds_per_product=max_rounds,
        initial_seller_price=150.0,  # Initial price offered by seller
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Define available products (similar to simple_negotiation.py structure)
    products = [
        {
            "name": "Premium Winter Jacket",
            "brand": "Mountain Gear",
            "price": 180.0,  # The product's own price
            "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
            "condition": "New",
            "material": "Gore-Tex",
        },
        {
            "name": "Running Shoes",
            "brand": "SportMax",
            "price": 120.0,  # The product's own price
            "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
            "condition": "New",
            "material": "Mesh and Synthetic",
        },
    ]
    
    # Initialize results dictionary and start time
    start_time = time.time()
    results = {
        "task": "Task1_multi_product_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_profile": user_profile,
        "product_results": [],
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    # First product negotiation (Winter Jacket)
    print("\n" + "="*60)
    print("Starting first product negotiation...")
    print("="*60)
    
    # Get user requirement
    # Use default requirement for automatic running
    user_requirement = "I need a high-quality winter jacket for cold weather"
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation...")
    print("="*60)
    
    # For first product, we can use a simple selection or default
    # Let's use the first product for the first negotiation
    selected_product = products[0]  # First product: Winter Jacket
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=selected_product,
        user_profile=user_profile,  # Pass user profile
        available_products=products,  # Pass all products to seller
    )
    
    # Start negotiation loop
    done = False
    
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
            
            # Collect first product results
            first_product_result = {
                "product_name": selected_product.get('name', 'Unknown'),
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "rounds": info.get('round', 0),
                "seller_reward": info.get('seller_reward'),
                "buyer_reward": info.get('buyer_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
            }
            results["product_results"].append(first_product_result)
            break
    
    # Continue with second product using default requirement
    # Use default requirement for automatic running
    user_requirement = "I need a high-quality pair of running shoes for jogging"
    print(f"Using default requirement for second product: {user_requirement}")
    
    # Select the most appropriate product based on user requirement
    print("\nSelecting the most appropriate product based on your requirement...")
    # Simple product selection: use the second product (Running Shoes) as default
    # ProductSelectorAgent has been removed, so we use a simple selection logic
    if "shoe" in user_requirement.lower() or "running" in user_requirement.lower() or "jogging" in user_requirement.lower():
        selected_product = products[1]  # Running Shoes
    else:
        selected_product = products[0]  # Default to first product
    print(f"Selected product: {selected_product['name']} (${selected_product['price']:.2f})")
    
    # Second product negotiation
    print("\n" + "="*60)
    print("Starting second product negotiation...")
    print("="*60)
    
    # Reset environment for new product (preserves context)
    print("\n" + "="*60)
    print("Starting new negotiation...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=selected_product,  # Selected product based on requirement
        user_profile=user_profile,  # Pass user profile
        clear_history=False,  # Preserve previous context
        available_products=products,  # Pass all products to seller
    )
    
    # Start negotiation loop
    done = False
    
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
            
            # Collect second product results
            second_product_result = {
                "product_name": selected_product.get('name', 'Unknown'),
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "rounds": info.get('round', 0),
                "seller_reward": info.get('seller_reward'),
                "buyer_reward": info.get('buyer_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
            }
            results["product_results"].append(second_product_result)
            break
    
    # Display final summary
    print("\n" + "="*60)
    print("All Products Negotiation Summary")
    print("="*60)
    
    final_info = env._get_info()
    if final_info.get("product_results"):
        print(f"\nTotal products negotiated: {len(final_info['product_results'])}")
        for i, result in enumerate(final_info["product_results"], 1):
            status_str = "✓ AGREED" if result["status"] == "agreed" else "✗ TIMEOUT"
            price_str = f"${result['agreed_price']:.2f}" if result.get('agreed_price') else "N/A"
            print(f"  {i}. {result['product_name']}: {status_str} @ {price_str} (Rounds: {result['rounds']})")
    else:
        print("No products were negotiated.")
    
    print("="*60)
    
    # Close environment
    env.close()
    print("\nMulti-product negotiation completed!")
    
    # Collect final summary
    final_info = env._get_info()
    if final_info.get("product_results"):
        results["product_results"] = final_info["product_results"]
    
    # Calculate overall success: all products must be successfully negotiated
    all_success = len(results["product_results"]) > 0 and all(
        p.get("success", False) for p in results["product_results"]
    )
    
    # Determine overall status based on product results
    # If any product has timeout/error, the overall status should reflect that
    has_timeout = any(p.get("status") == "timeout" for p in results["product_results"])
    has_error = any(p.get("error") is not None for p in results["product_results"])
    
    if has_error:
        overall_status = "error"
    elif has_timeout:
        overall_status = "timeout"
    elif all_success:
        overall_status = "agreed"
    else:
        overall_status = "completed"
    
    # Update results with final status
    results.update({
        "status": overall_status,
        "success": all_success,
        "total_products": len(results["product_results"]),
        "elapsed_time": time.time() - start_time,
        "buyer_max_price": buyer_max_price,
        "seller_min_price": seller_min_price,
        "model": get_model_name(model),
    })
    
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
        output_file = run_dir / "Task1_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task1: Multi-Product Negotiation Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Total Products Negotiated: {results['total_products']}\n")
            elapsed_time = results.get('elapsed_time', 0)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            f.write("Product Results:\n")
            for i, result in enumerate(results["product_results"], 1):
                status_str = "✓ AGREED" if result.get("status") == "agreed" else "✗ TIMEOUT"
                price_str = f"${result['agreed_price']:.2f}" if result.get('agreed_price') else "N/A"
                f.write(f"  {i}. {result.get('product_name', 'Unknown')}: {status_str} @ {price_str} (Rounds: {result.get('rounds', 0)})\n")
            f.write("\n")
            if results.get('error'):
                f.write(f"Error: {results['error']}\n")
        
        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task1: Multi-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

