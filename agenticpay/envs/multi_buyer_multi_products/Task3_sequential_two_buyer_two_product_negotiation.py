"""Task3 Sequential Two-Buyer Two-Product Negotiation Environment Implementation

Supports sequential negotiation where seller chooses one buyer per round to negotiate with.
Seller can switch between two buyers and make a deal with either buyer.
Prices represent total price for both products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task3SequentialTwoBuyerTwoProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Buyer Two-Product Negotiation Environment
    
    Manages sequential negotiation process where seller chooses one buyer per round to negotiate with.
    Seller can switch between two buyers and make a deal with either buyer.
    Prices represent total price for both products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
        gamma: float = 0.99,
        deal_score_weight: float = 30.0,
        quality_score_weight: float = 55.0,
        efficiency_score_weight: float = 15.0,
        failure_penalty_weight: float = 15.0,
        buyer_deal_weight: float = 30.0,
        buyer_utility_weight: float = 55.0,
        buyer_efficiency_weight: float = 15.0,
        buyer_failure_penalty_weight: float = 15.0,
        seller_deal_weight: float = 30.0,
        seller_utility_weight: float = 55.0,
        seller_efficiency_weight: float = 15.0,
        seller_failure_penalty_weight: float = 15.0,
    ):
        """Initialize sequential multi-buyer multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial total price offered by seller for both products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for both products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for both products)
            seller_min_price: Minimum acceptable total price for seller (confidential, for both products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
            gamma: Discount factor for GlobalScore calculation, controls penalty for longer negotiations (default: 0.99, range: 0.97-0.995)
            deal_score_weight: Weight D for DealScore component (default: 30.0)
            quality_score_weight: Weight W for QualityScore component (default: 55.0)
            efficiency_score_weight: Weight E for EfficiencyScore component (default: 15.0)
            failure_penalty_weight: Weight F for FailurePenalty component (default: 15.0)
            buyer_deal_weight: Weight Db for Buyer Deal Bonus (default: 30.0)
            buyer_utility_weight: Weight Wb for Buyer utility component (default: 55.0)
            buyer_efficiency_weight: Weight Eb for Buyer Efficiency Bonus (default: 15.0)
            buyer_failure_penalty_weight: Weight Fb for Buyer Failure Penalty (default: 15.0)
            seller_deal_weight: Weight Ds for Seller Deal Bonus (default: 30.0)
            seller_utility_weight: Weight Ws for Seller utility component (default: 55.0)
            seller_efficiency_weight: Weight Es for Seller Efficiency Bonus (default: 15.0)
            seller_failure_penalty_weight: Weight Fs for Seller Failure Penalty (default: 15.0)
        """
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        
        # Set default reward weights
        default_weights = {
            "buyer_savings": 1.0,      # Buyer savings weight
            "seller_profit": 1.0,      # Seller profit weight
            "time_cost": 0.1,          # Time cost weight (reduced impact)
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
        # Score calculation parameters
        self.gamma = gamma
        self.deal_score_weight = deal_score_weight  # D
        self.quality_score_weight = quality_score_weight  # W
        self.efficiency_score_weight = efficiency_score_weight  # E
        self.failure_penalty_weight = failure_penalty_weight  # F
        # Buyer score weights
        self.buyer_deal_weight = buyer_deal_weight  # Db
        self.buyer_utility_weight = buyer_utility_weight  # Wb
        self.buyer_efficiency_weight = buyer_efficiency_weight  # Eb
        self.buyer_failure_penalty_weight = buyer_failure_penalty_weight  # Fb
        # Seller score weights
        self.seller_deal_weight = seller_deal_weight  # Ds
        self.seller_utility_weight = seller_utility_weight  # Ws
        self.seller_efficiency_weight = seller_efficiency_weight  # Es
        self.seller_failure_penalty_weight = seller_failure_penalty_weight  # Fs
        
        # Call parent class initialization
        super().__init__()
        
        # State management - separate for each buyer
        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which buyer is currently selected and which buyer was chosen for the deal
        self.current_selected_buyer: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_deal_price: Optional[float] = None
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description (should describe purchasing two products)
            product_info: Product information containing two products and their prices
                Expected format: {
                    "products": [
                        {"name": "Product1", "price": 100.0, ...},
                        {"name": "Product2", "price": 80.0, ...}
                    ]
                }
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.current_selected_buyer = None
        self.final_selected_buyer = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Extract product_images for VLM (from image_url or image_path in each product)
        product_images: Optional[List[str]] = []
        for p in products:
            img_url = p.get("image_path") or p.get("image_url")
            if img_url:
                product_images.append(img_url)
        if not product_images:
            product_images = None
        self.product_images = product_images
        
        # Calculate total price of both products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        
        # Initialize Buyer1 Agent
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "buyer_id": 1,  # Identify as buyer 1
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "buyer_id": 2,  # Identify as buyer 2
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller Agent (seller knows about both buyers and both products)
        seller_context = {
            "product_info": self.product_info,  # Seller can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "initial_price": self.initial_seller_price,  # Initial total price
            "min_price": self.seller_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "num_buyers": 2,  # Inform seller there are 2 buyers
            "negotiation_mode": "sequential",  # Inform seller this is sequential negotiation
        }
        self.seller_agent.initialize(seller_context)
        
        # No initial seller offer - negotiation starts with buyers' first messages
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        selected_buyer: int,  # 1 or 2, which buyer seller chooses to negotiate with this round
        seller_action: Optional[str] = None,
        buyer_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, seller chooses one buyer to negotiate with, then buyer and seller exchange messages.
        Order: buyer -> seller (buyer responds first, then seller can see buyer's message)
        Prices represent total price for both products.
        
        Args:
            selected_buyer: Which buyer (1 or 2) seller chooses to negotiate with this round
            seller_action: Seller's response (optional)
            buyer_action: Selected buyer's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_buyer not in [1, 2]:
            raise ValueError(f"selected_buyer must be 1 or 2, got {selected_buyer}")
        
        self.current_selected_buyer = selected_buyer
        
        # Add messages to memory in order: buyer -> seller
        # Process buyer action first
        if buyer_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer1.update(buyer_price=buyer_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer2.update(buyer_price=buyer_price)
        
        # Process seller action after buyer (seller can see buyer's message)
        if seller_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer1.update(seller_price=seller_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected buyer
        # Deal is made when: (1) price difference <= tolerance, or (2) seller's offer <= buyer's offer
        if selected_buyer == 1:
            if (buyer_action is not None and 
                self.state_buyer1.buyer_price is not None and 
                self.state_buyer1.seller_price is not None):
                price_diff = abs(self.state_buyer1.buyer_price - self.state_buyer1.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 1
                    self.final_deal_price = (self.state_buyer1.buyer_price + self.state_buyer1.seller_price) / 2
                elif self.state_buyer1.seller_price <= self.state_buyer1.buyer_price:
                    self.final_selected_buyer = 1
                    self.final_deal_price = self.state_buyer1.seller_price
        else:  # selected_buyer == 2
            if (buyer_action is not None and 
                self.state_buyer2.buyer_price is not None and 
                self.state_buyer2.seller_price is not None):
                price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 2
                    self.final_deal_price = (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
                elif self.state_buyer2.seller_price <= self.state_buyer2.buyer_price:
                    self.final_selected_buyer = 2
                    self.final_deal_price = self.state_buyer2.seller_price
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected buyer in this round (sequential negotiation)
        step_buyer1_reward = self._calculate_step_buyer_reward(1) if selected_buyer == 1 else None
        step_buyer2_reward = self._calculate_step_buyer_reward(2) if selected_buyer == 2 else None
        step_seller_reward = self._calculate_step_seller_reward()
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        if step_buyer1_reward is not None:
            info["step_buyer1_reward"] = step_buyer1_reward
        if step_buyer2_reward is not None:
            info["step_buyer2_reward"] = step_buyer2_reward
        info["step_seller_reward"] = step_seller_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller_reward"] = seller_reward
            # Calculate GlobalScore, BuyerScore, and SellerScore for final result
            # Note: current_round has been incremented to reflect the completed round
            # Don't print here - will be printed in render() after Round Summary
            global_score = self._calculate_global_score(print_details=False)
            info["global_score"] = global_score
            buyer_score = self._calculate_buyer_score(print_details=False)
            info["buyer_score"] = buyer_score
            seller_score = self._calculate_seller_score(print_details=False)
            info["seller_score"] = seller_score
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays seller and buyer outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for both products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history_buyer1 = self.memory_buyer1.get_history()
        history_buyer2 = self.memory_buyer2.get_history()
        
        # Determine which round's messages to display
        # Messages are stored with the round value at the time of storage (before current_round is incremented)
        # In step(), messages are added first, then current_round is incremented
        # So for any completed round, messages are stored at current_round - 1
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        # Display round number: current_round is already incremented, so it represents the completed round number
        display_round = self.current_round
        
        # Display product info
        if self.product_info:
            products = self.product_info.get("products", [])
            if products:
                output_lines.append(f"\n{'='*60}")
                output_lines.append("Products:")
                for i, p in enumerate(products, 1):
                    name = p.get("name", f"Product {i}")
                    price = p.get("price", 0.0)
                    output_lines.append(f"  {i}. {name}: ${price:.2f}")
                total_price = sum(p.get("price", 0.0) for p in products)
                output_lines.append(f"  Total Product Price: ${total_price:.2f}")
                output_lines.append(f"{'='*60}")
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which buyer was selected this round
        if self.current_selected_buyer is not None:
            output_lines.append(f"\n[Selected Buyer: Buyer {self.current_selected_buyer}]")
        
        # Display Buyer1 conversation (if this round negotiated with buyer1)
        if self.current_selected_buyer == 1 and history_buyer1:
            round_messages_b1 = [
                msg for msg in history_buyer1 if msg["round"] == round_to_display
            ]
            if round_messages_b1:
                output_lines.append(f"\n[BUYER 1 Conversation]:")
                # Display buyer message first (if exists)
                buyer_msg_b1 = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_b1:
                    output_lines.append(f"  [BUYER]: {buyer_msg_b1['content']}")
                
                # Display seller message (if exists)
                seller_msg_b1 = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_b1:
                    output_lines.append(f"  [SELLER]: {seller_msg_b1['content']}")
        
        # Display Buyer2 conversation (if this round negotiated with buyer2)
        if self.current_selected_buyer == 2 and history_buyer2:
            round_messages_b2 = [
                msg for msg in history_buyer2 if msg["round"] == round_to_display
            ]
            if round_messages_b2:
                output_lines.append(f"\n[BUYER 2 Conversation]:")
                # Display buyer message first (if exists)
                buyer_msg_b2 = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_b2:
                    output_lines.append(f"  [BUYER]: {buyer_msg_b2['content']}")
                
                # Display seller message (if exists)
                seller_msg_b2 = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_b2:
                    output_lines.append(f"  [SELLER]: {seller_msg_b2['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {display_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1 prices (total for both products)
        output_lines.append(f"\nBuyer 1:")
        if self.state_buyer1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_buyer1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_buyer1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_buyer1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2 prices (total for both products)
        output_lines.append(f"\nBuyer 2:")
        if self.state_buyer2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_buyer2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_buyer2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_buyer2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.final_selected_buyer is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Buyer {self.final_selected_buyer}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Total Price: ${self.final_deal_price:.2f}")
        else:
            output_lines.append(f"\n  ✗ NO DEAL YET")
        
        # Display negotiation status
        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout"
        }
        output_lines.append(f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}")
        
        output_lines.append(f"{'='*60}\n")
        
        output = "\n".join(output_lines)
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history_buyer1": self.memory_buyer1.get_history(),
            "conversation_history_buyer2": self.memory_buyer2.get_history(),
            "current_round": self.current_round,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,  # Total price for both products
            "seller_price_buyer1": self.state_buyer1.seller_price,  # Total price for both products
            "buyer2_price": self.state_buyer2.buyer_price,  # Total price for both products
            "seller_price_buyer2": self.state_buyer2.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
            "product_info": self.product_info,
        }
        # Include product_images for VLM (agent passes img to model when is_vlm)
        if getattr(self, "product_images", None) is not None:
            obs["product_images"] = self.product_images
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,  # Total price for both products
            "seller_price_buyer1": self.state_buyer1.seller_price,  # Total price for both products
            "buyer2_price": self.state_buyer2.buyer_price,  # Total price for both products
            "seller_price_buyer2": self.state_buyer2.seller_price,  # Total price for both products
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,
        }
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text
        
        Priority: 
        1. Extract from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format (preferred)
        2. Fall back to ### $X ### format
        3. Fall back to other price patterns
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price, returns None if not found
        """
        def parse_price(price_str: str) -> Optional[float]:
            """Parse price string, removing commas and converting to float"""
            try:
                # Remove commas from price string (e.g., "8,750" -> "8750")
                cleaned = price_str.replace(',', '')
                price = float(cleaned)
                if price > 0:
                    return price
            except ValueError:
                pass
            return None
        
        # Priority 1: Extract price from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format
        # Matches: ### BUYER_PRICE($100.50) ###, ### SELLER_PRICE($150) ###, ### BUYER_PRICE($8,750) ###, etc.
        labeled_price_pattern = r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###'
        matches = re.findall(labeled_price_pattern, text, re.IGNORECASE)
        if matches:
            price = parse_price(matches[-1])  # Take the last match
            if price is not None:
                return price
        
        # Priority 2: Extract price from ### $X ### format (backward compatibility)
        # Matches: ### $100.50 ###, ### $100 ###, ###$120###, ### $8,750 ###, etc.
        triple_hash_pattern = r'###\s*\$([\d,]+\.?\d*)\s*###'
        matches = re.findall(triple_hash_pattern, text, re.IGNORECASE)
        if matches:
            price = parse_price(matches[-1])  # Take the last match
            if price is not None:
                return price
        
        # Priority 3: Fall back to other price patterns
        fallback_patterns = [
            r'\$([\d,]+\.?\d*)',  # $100.50 or $100 or $8,750
            r'([\d,]+\.?\d*)\s*dollars?',  # 100.50 dollars or 8,750 dollars
            r'([\d,]+\.?\d*)\s*USD',  # 100.50 USD or 8,750 USD
            r'price.*?([\d,]+\.?\d*)',  # price 100.50 or price 8,750
            r'offer.*?([\d,]+\.?\d*)',  # offer 100.50 or offer 8,750
            r'total.*?(\d+\.?\d*)',  # total 100.50
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                price = parse_price(matches[-1])  # Take the last match
                if price is not None:
                    return price
        
        return None
    
    def _check_make_deal(self, text: str) -> bool:
        """Check if buyer wants to make a deal
        
        Args:
            text: Buyer's response text
            
        Returns:
            Whether buyer wants to make a deal
        """
        # First check for the fixed format "MAKE_DEAL"
        if 'MAKE_DEAL' in text.upper():
            return True
        
        # More specific patterns to avoid false positives
        # Exclude phrases like "I hope we can reach an agreement" which are just expressions of hope
        make_deal_patterns = [
            r'\bi\s+accept\b',  # "I accept"
            r'\bi\s+agree\b',  # "I agree"
            r'\baccept\s+your\s+offer\b',  # "accept your offer"
            r'\bagree\s+to\s+the\s+deal\b',  # "agree to the deal"
            r'\bmake\s+a\s+deal\b',  # "make a deal"
            r'\bwe\s+have\s+a\s+deal\b',  # "we have a deal"
            r'\blet\'?s\s+do\s+it\b',  # "let's do it"
            r'\bi\'?ll\s+take\s+it\b',  # "I'll take it"
            r'\bi\'?m\s+accepting\b',  # "I'm accepting"
            r'\bi\'?m\s+agreeing\b',  # "I'm agreeing"
            r'\bdeal\s*!',  # "deal!"
            r'\bi\s+accept\s+the\s+price\b',  # "I accept the price"
        ]
        
        text_lower = text.lower()
        
        # Exclude common false positive patterns
        false_positive_patterns = [
            r'hope\s+.*\s+agreement',  # "hope ... agreement"
            r'hoping\s+.*\s+agreement',  # "hoping ... agreement"
            r'would\s+like\s+.*\s+agreement',  # "would like ... agreement"
            r'looking\s+forward\s+.*\s+agreement',  # "looking forward ... agreement"
        ]
        
        # Check for false positives first
        for pattern in false_positive_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for actual deal patterns
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate global reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a buyer, use that buyer's max_price for calculation.
        Prices represent total price for both products.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if selected_buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        Prices represent total price for both products.
        
        If deal is reached with this buyer:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another buyer:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this buyer
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            # Deal reached with this buyer: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Get this buyer's max_price
            buyer_max_price = None
            if buyer_id == 1:
                buyer_max_price = self.buyer1_max_price
            elif buyer_id == 2:
                buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{buyer_id}_max={buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another buyer: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this buyer)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        Prices represent total price for both products.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round.
        This is calculated every round, not just at the end.
        Prices represent total price for both products.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer for both products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer state
        buyer_state = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_state = self.state_buyer1
            buyer_max_price = self.buyer1_max_price
        elif buyer_id == 2:
            buyer_state = self.state_buyer2
            buyer_max_price = self.buyer2_max_price
        
        # Calculate buyer savings from current offer: buyer_max_price - buyer_price (for both products)
        if buyer_state is not None and buyer_state.buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_state.buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _get_selected_buyer_max_price(self) -> Optional[float]:
        """Get the final selected buyer's max_price
        
        Returns:
            Final selected buyer's max_price, or None if no buyer is selected
        """
        if self.final_selected_buyer == 1:
            return self.buyer1_max_price
        elif self.final_selected_buyer == 2:
            return self.buyer2_max_price
        return None
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Uses the final selected buyer's max_price and seller's min_price for calculation.
        If no buyer is selected, calculates failure penalty.
        
        Let:
        - buyer_max_price = maximum price the final selected buyer is willing to pay
        - seller_min_price = minimum price the seller is willing to accept
        - Z = buyer_max_price - seller_min_price
        - γ (gamma) controls how strongly longer negotiations are penalized (default: 0.99)
        
        valid_range = (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        feasible_deal = negotiation reached agreement
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            u_b = (buyer_max_price - p) / Z          # in [0, 1]
            u_s = (p - seller_min_price) / Z         # in [0, 1]
            Q = 4 * u_b * u_s                        # in [0, 1]
            
            DealScore       = D * discount
            QualityScore    = W * Q * discount
            EfficiencyScore = E * discount
            
            GlobalScore = DealScore + QualityScore + EfficiencyScore
        Else:
            FailurePenalty = -F * (1 - discount)
            GlobalScore = FailurePenalty
        
        Settings:
            D = deal_score_weight (default: 30)
            W = quality_score_weight (default: 55)
            E = efficiency_score_weight (default: 15)
            F = failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
            T = max_rounds
        
        Returns:
            GlobalScore value (only calculated at final result)
        """
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        seller_min_price = self.seller_min_price
        
        # Check if we have required prices
        if selected_buyer_max_price is None or seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  selected_buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Calculate Z
        Z = selected_buyer_max_price - seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (selected_buyer_max_price - final_price) / Z
            u_s = (final_price - seller_min_price) / Z
            
            # Calculate Q = 4 * u_b * u_s (in [0,1])
            Q = 4.0 * u_b * u_s
            
            # Calculate component scores
            deal_score = self.deal_score_weight * discount  # D * discount
            quality_score = self.quality_score_weight * Q * discount  # W * Q * discount
            efficiency_score = self.efficiency_score_weight * discount  # E * discount
            
            # Calculate GlobalScore
            global_score = deal_score + quality_score + efficiency_score
            
            if print_details:
                # Debug output header
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (selected_buyer_max_price({selected_buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  Q = 4 * u_b({u_b:.4f}) * u_s({u_s:.4f}) = {Q:.4f}")
                print(f"  DealScore = D({self.deal_score_weight:.1f}) * discount({discount:.6f}) = {deal_score:.3f}")
                print(f"  QualityScore = W({self.quality_score_weight:.1f}) * Q({Q:.4f}) * discount({discount:.6f}) = {quality_score:.3f}")
                print(f"  EfficiencyScore = E({self.efficiency_score_weight:.1f}) * discount({discount:.6f}) = {efficiency_score:.3f}")
                print(f"  GlobalScore = DealScore({deal_score:.3f}) + QualityScore({quality_score:.3f}) + EfficiencyScore({efficiency_score:.3f}) = {global_score:.3f}")
            
            return global_score
        else:
            # Calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore based on the formula
        
        Uses the final selected buyer's max_price and seller's min_price for calculation.
        
        u_b = (buyer_max_price - p) / (buyer_max_price - seller_min_price)
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            BuyerScore = discount * (Db + Wb * u_b + Eb)
        Else:
            BuyerScore = -Fb * (1 - discount)
        
        Settings:
            Db = buyer_deal_weight (default: 30)
            Wb = buyer_utility_weight (default: 55)
            Eb = buyer_efficiency_weight (default: 15)
            Fb = buyer_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            BuyerScore value (only calculated at final result)
        """
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        seller_min_price = self.seller_min_price
        
        # Check if we have required prices
        if selected_buyer_max_price is None or seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  selected_buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Calculate Z
        Z = selected_buyer_max_price - seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (selected_buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (selected_buyer_max_price({selected_buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  BuyerScore = discount({discount:.6f}) * (Db({self.buyer_deal_weight:.1f}) + Wb({self.buyer_utility_weight:.1f}) * u_b({u_b:.4f}) + Eb({self.buyer_efficiency_weight:.1f}))")
                print(f"  BuyerScore = {discount:.6f} * ({self.buyer_deal_weight:.1f} + {self.buyer_utility_weight * u_b:.4f} + {self.buyer_efficiency_weight:.1f}) = {buyer_score:.3f}")
            
            return buyer_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
        Uses the final selected buyer's max_price and seller's min_price for calculation.
        If no buyer is selected or deal not reached, calculates failure penalty.
        
        u_s = (p - seller_min_price) / (buyer_max_price - seller_min_price)
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            SellerScore = discount * (Ds + Ws * u_s + Es)
        Else:
            SellerScore = -Fs * (1 - discount)
        
        Settings:
            Ds = seller_deal_weight (default: 30)
            Ws = seller_utility_weight (default: 55)
            Es = seller_efficiency_weight (default: 15)
            Fs = seller_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            SellerScore value (only calculated at final result)
        """
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        seller_min_price = self.seller_min_price
        
        # Check if we have required prices
        if selected_buyer_max_price is None or seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  selected_buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Calculate Z
        Z = selected_buyer_max_price - seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f}) * u_s({u_s:.4f}) + Es({self.seller_efficiency_weight:.1f}))")
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - seller_min_price({seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            
            return seller_score
    
    def _calculate_step_seller_reward(self) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        Prices represent total price for both products.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer for both products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller price from the selected buyer
        seller_price = None
        if self.current_selected_buyer == 1:
            seller_price = self.state_buyer1.seller_price
        elif self.current_selected_buyer == 2:
            seller_price = self.state_buyer2.seller_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price (for both products)
        if seller_price is not None and self.seller_min_price is not None:
            seller_profit = seller_price - self.seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _print_global_score_details(self):
        """Print GlobalScore calculation details (called from example after Round Summary)"""
        self._calculate_global_score(print_details=True)
    
    def _print_buyer_score_details(self):
        """Print BuyerScore calculation details (called from example after Round Summary)"""
        self._calculate_buyer_score(print_details=True)
    
    def _print_seller_score_details(self):
        """Print SellerScore calculation details (called from example after Round Summary)"""
        self._calculate_seller_score(print_details=True)

