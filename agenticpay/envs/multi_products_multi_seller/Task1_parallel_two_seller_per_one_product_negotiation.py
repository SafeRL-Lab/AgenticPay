"""Task1 Parallel Two-Seller Per One Product Negotiation Environment Implementation

Supports parallel negotiation between one buyer and two sellers, where each seller
has their own unique product. Buyer negotiates with both sellers simultaneously
and can choose to make a deal with the seller offering the lower price.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task1ParallelTwoSellerPerOneProductNegotiation(BaseEnv):
    """Task1 Parallel Two-Seller Per One Product Negotiation Environment
    
    Manages parallel negotiation process between one buyer and two sellers, where
    each seller has their own unique product. Buyer negotiates with both sellers
    simultaneously and can choose to make a deal with the lower price.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 100.0,
        initial_seller2_price: float = 110.0,
        buyer_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        buyer_reward_aggregation: str = "average",
        seller_reward_aggregation: str = "average",
        gamma: float = 0.99,
        deal_score_weight: float = 10.0,
        quality_score_weight: float = 80.0,
        efficiency_score_weight: float = 10.0,
        failure_penalty_weight: float = 15.0,
        buyer_deal_weight: float = 10.0,
        buyer_utility_weight: float = 80.0,
        buyer_efficiency_weight: float = 10.0,
        buyer_failure_penalty_weight: float = 15.0,
        seller_deal_weight: float = 10.0,
        seller_utility_weight: float = 80.0,
        seller_efficiency_weight: float = 10.0,
        seller_failure_penalty_weight: float = 15.0,
    ):
        """Initialize multi-seller negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial price offered by seller1
            initial_seller2_price: Initial price offered by seller2
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller1_min_price: Minimum acceptable price for seller1 (confidential)
            seller2_min_price: Minimum acceptable price for seller2 (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
            buyer_reward_aggregation: How to aggregate buyer rewards across sellers.
                Options: "average", "max", "min" (default: "average")
            seller_reward_aggregation: How to aggregate seller rewards across buyers.
                Options: "average", "max", "min" (default: "average")
            gamma: Discount factor for GlobalScore calculation, controls penalty for longer negotiations (default: 0.99, range: 0.97-0.995)
            deal_score_weight: Weight D for DealScore component (default: 10.0)
            quality_score_weight: Weight W for QualityScore component (default: 80.0)
            efficiency_score_weight: Weight E for EfficiencyScore component (default: 10.0)
            failure_penalty_weight: Weight F for FailurePenalty component (default: 15.0)
            buyer_deal_weight: Weight Db for Buyer Deal Bonus (default: 10.0)
            buyer_utility_weight: Weight Wb for Buyer utility component (default: 80.0)
            buyer_efficiency_weight: Weight Eb for Buyer Efficiency Bonus (default: 10.0)
            buyer_failure_penalty_weight: Weight Fb for Buyer Failure Penalty (default: 15.0)
            seller_deal_weight: Weight Ds for Seller Deal Bonus (default: 10.0)
            seller_utility_weight: Weight Ws for Seller utility component (default: 80.0)
            seller_efficiency_weight: Weight Es for Seller Efficiency Bonus (default: 10.0)
            seller_failure_penalty_weight: Weight Fs for Seller Failure Penalty (default: 15.0)
        """
        self.buyer_agent = buyer_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.buyer_max_price = buyer_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
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
        
        # Set reward aggregation methods
        self.buyer_reward_aggregation = buyer_reward_aggregation
        self.seller_reward_aggregation = seller_reward_aggregation
        
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
        
        # State management - separate for each seller
        self.memory_seller1 = ConversationMemory()
        self.memory_seller2 = ConversationMemory()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which seller was chosen for the deal
        self.selected_seller: Optional[int] = None  # 1 or 2
        self.final_deal_price: Optional[float] = None
        
        # Store product info for each seller
        self.seller1_product_info: Optional[Dict[str, Any]] = None
        self.seller2_product_info: Optional[Dict[str, Any]] = None
    
    def reset(
        self,
        user_requirement: str = "",
        seller1_product_info: Optional[Dict[str, Any]] = None,
        seller2_product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description
            seller1_product_info: Product information for seller1
            seller2_product_info: Product information for seller2
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.selected_seller = None
        self.final_deal_price = None
        
        # Store product info for each seller
        self.seller1_product_info = seller1_product_info or {}
        self.seller2_product_info = seller2_product_info or {}
        
        # Initialize Buyer Agent (buyer knows about both sellers and their products)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
        }
        self.buyer_agent.initialize(buyer_context)
        
        # Initialize Seller1 Agent with its own product
        seller1_context = {
            "product_info": self.seller1_product_info,
            "initial_price": self.initial_seller1_price,
            "min_price": self.seller1_min_price,
            "environment_info": self.environment_info,
            "seller_id": 1,  # Identify as seller 1
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent with its own product
        seller2_context = {
            "product_info": self.seller2_product_info,
            "initial_price": self.initial_seller2_price,
            "min_price": self.seller2_min_price,
            "environment_info": self.environment_info,
            "seller_id": 2,  # Identify as seller 2
        }
        self.seller2_agent.initialize(seller2_context)
        
        # No initial seller offers - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer_action_seller1: Optional[str] = None,
        buyer_action_seller2: Optional[str] = None,
        seller1_action: Optional[str] = None,
        seller2_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, buyer responds to both sellers first, then both sellers respond.
        Order: buyer -> seller
        Buyer can choose to make a deal with either seller.
        
        Args:
            buyer_action_seller1: Buyer's response to seller1 (optional)
            buyer_action_seller2: Buyer's response to seller2 (optional)
            seller1_action: Seller1's response (optional)
            seller2_action: Seller2's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory in order: buyer -> seller
        # Process buyer actions first (seller1 conversation)
        if buyer_action_seller1 is not None:
            self.memory_seller1.add_message("buyer", buyer_action_seller1, self.current_round)
            buyer_price_seller1 = self._extract_price(buyer_action_seller1)
            if buyer_price_seller1 is not None:
                self.state_seller1.update(buyer_price=buyer_price_seller1)
        
        # Process buyer actions (seller2 conversation)
        if buyer_action_seller2 is not None:
            self.memory_seller2.add_message("buyer", buyer_action_seller2, self.current_round)
            buyer_price_seller2 = self._extract_price(buyer_action_seller2)
            if buyer_price_seller2 is not None:
                self.state_seller2.update(buyer_price=buyer_price_seller2)
        
        # Process seller actions after buyer (seller1 conversation)
        if seller1_action is not None:
            self.memory_seller1.add_message("seller", seller1_action, self.current_round)
            seller1_price = self._extract_price(seller1_action)
            if seller1_price is not None:
                self.state_seller1.update(seller_price=seller1_price)
        
        # Process seller actions after buyer (seller2 conversation)
        if seller2_action is not None:
            self.memory_seller2.add_message("seller", seller2_action, self.current_round)
            seller2_price = self._extract_price(seller2_action)
            if seller2_price is not None:
                self.state_seller2.update(seller_price=seller2_price)
        
        # After processing all actions, check if buyer wants to make a deal
        # Deal can be reached when: (1) price difference <= tolerance, or (2) seller's offer <= buyer's offer
        # Check if seller1 can make deal
        can_make_deal_seller1 = False
        if (buyer_action_seller1 is not None and 
            self.state_seller1.buyer_price is not None and 
            self.state_seller1.seller_price is not None):
            price_diff = abs(self.state_seller1.buyer_price - self.state_seller1.seller_price)
            if price_diff <= self.price_tolerance:
                can_make_deal_seller1 = True
            elif self.state_seller1.seller_price <= self.state_seller1.buyer_price:
                can_make_deal_seller1 = True
        
        # Check if seller2 can make deal
        can_make_deal_seller2 = False
        if (buyer_action_seller2 is not None and 
            self.state_seller2.buyer_price is not None and 
            self.state_seller2.seller_price is not None):
            price_diff = abs(self.state_seller2.buyer_price - self.state_seller2.seller_price)
            if price_diff <= self.price_tolerance:
                can_make_deal_seller2 = True
            elif self.state_seller2.seller_price <= self.state_seller2.buyer_price:
                can_make_deal_seller2 = True
        
        if can_make_deal_seller1 or can_make_deal_seller2:
            # Buyer wants to make a deal and price tolerance is satisfied, choose the seller with lower price
            price1 = self._get_effective_price_seller1() if can_make_deal_seller1 else None
            price2 = self._get_effective_price_seller2() if can_make_deal_seller2 else None
            
            if price1 is not None and price2 is not None:
                # Both prices available, choose the lower one
                if price1 <= price2:
                    self.selected_seller = 1
                    if self.state_seller1.seller_price <= self.state_seller1.buyer_price:
                        self.final_deal_price = self.state_seller1.seller_price
                    else:
                        self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
                else:
                    self.selected_seller = 2
                    if self.state_seller2.seller_price <= self.state_seller2.buyer_price:
                        self.final_deal_price = self.state_seller2.seller_price
                    else:
                        self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
            elif price1 is not None:
                self.selected_seller = 1
                if self.state_seller1.seller_price <= self.state_seller1.buyer_price:
                    self.final_deal_price = self.state_seller1.seller_price
                else:
                    self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
            elif price2 is not None:
                self.selected_seller = 2
                if self.state_seller2.seller_price <= self.state_seller2.buyer_price:
                    self.final_deal_price = self.state_seller2.seller_price
                else:
                    self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
        
        # Check if deal is made (buyer chose a seller)
        terminated = False
        truncated = False
        reward = 0.0
        buyer_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        step_buyer_reward = self._calculate_step_buyer_reward()
        step_seller1_reward = self._calculate_step_seller_reward(1)
        step_seller2_reward = self._calculate_step_seller_reward(2)
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer_reward"] = step_buyer_reward
        info["step_seller1_reward"] = step_seller1_reward
        info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_seller"] = self.selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer_reward"] = buyer_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            # Calculate GlobalScore, BuyerScore, and SellerScore for final result
            # Note: current_round has been incremented to reflect the completed round
            # Don't print here - will be printed in example code after Step Rewards
            global_score = self._calculate_global_score(print_details=False)
            info["global_score"] = global_score
            buyer_score = self._calculate_buyer_score(print_details=False)
            info["buyer_score"] = buyer_score
            seller_score = self._calculate_seller_score(print_details=False)
            info["seller_score"] = seller_score
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        
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
        history_seller1 = self.memory_seller1.get_history()
        history_seller2 = self.memory_seller2.get_history()
        
        # Determine which round's messages to display
        # Messages are stored with the round value at the time of storage (before current_round is incremented)
        # In step(), messages are added first, then current_round is incremented
        # So for any completed round, messages are stored at current_round - 1
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        
        # Display round number: current_round is already incremented, so it represents the completed round number
        display_round = self.current_round
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Parallel Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display Seller1 conversation
        output_lines.append(f"\n[SELLER 1 Conversation]:")
        if history_seller1:
            round_messages_s1 = [
                msg for msg in history_seller1 if msg["round"] == round_to_display
            ]
            if round_messages_s1:
                # Display buyer message first (if exists)
                buyer_msg_s1 = next(
                    (msg for msg in round_messages_s1 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_s1:
                    output_lines.append(f"  [BUYER]: {buyer_msg_s1['content']}")
                
                # Display seller message (if exists)
                seller_msg_s1 = next(
                    (msg for msg in round_messages_s1 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_s1:
                    output_lines.append(f"  [SELLER]: {seller_msg_s1['content']}")
        
        # Display Seller2 conversation
        output_lines.append(f"\n[SELLER 2 Conversation]:")
        if history_seller2:
            round_messages_s2 = [
                msg for msg in history_seller2 if msg["round"] == round_to_display
            ]
            if round_messages_s2:
                # Display buyer message first (if exists)
                buyer_msg_s2 = next(
                    (msg for msg in round_messages_s2 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_s2:
                    output_lines.append(f"  [BUYER]: {buyer_msg_s2['content']}")
                
                # Display seller message (if exists)
                seller_msg_s2 = next(
                    (msg for msg in round_messages_s2 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_s2:
                    output_lines.append(f"  [SELLER]: {seller_msg_s2['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Seller1 prices
        output_lines.append(f"\nSeller 1:")
        if self.state_seller1.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_seller1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_seller1.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_seller1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display Seller2 prices
        output_lines.append(f"\nSeller 2:")
        if self.state_seller2.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_seller2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_seller2.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_seller2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display deal status
        if self.selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Seller {self.selected_seller}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Price: ${self.final_deal_price:.2f}")
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
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_seller1": self.memory_seller1.get_history(),
            "conversation_history_seller2": self.memory_seller2.get_history(),
            "current_round": self.current_round,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "status": self.negotiation_info.status.value,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
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
        
        # Also check for other make deal patterns
        make_deal_patterns = [
            r'make\s+deal',
            r'accept',
            r'agree',
            r'deal',
            r'let\'?s\s+do\s+it',
            r'i\'?ll\s+take\s+it',
            r'we\s+have\s+a\s+deal',
        ]
        
        text_lower = text.lower()
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _get_effective_price_seller1(self) -> Optional[float]:
        """Get effective price for seller1 (deal price when deal is reachable, otherwise seller price)
        
        When seller's offer <= buyer's offer: effective price = seller_price.
        When within tolerance: effective price = (buyer_price + seller_price) / 2.
        
        Returns:
            Effective price for seller1
        """
        if self.state_seller1.buyer_price is not None and self.state_seller1.seller_price is not None:
            if self.state_seller1.seller_price <= self.state_seller1.buyer_price:
                return self.state_seller1.seller_price
            price_diff = abs(self.state_seller1.buyer_price - self.state_seller1.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
        return self.state_seller1.seller_price
    
    def _get_effective_price_seller2(self) -> Optional[float]:
        """Get effective price for seller2 (deal price when deal is reachable, otherwise seller price)
        
        When seller's offer <= buyer's offer: effective price = seller_price.
        When within tolerance: effective price = (buyer_price + seller_price) / 2.
        
        Returns:
            Effective price for seller2
        """
        if self.state_seller2.buyer_price is not None and self.state_seller2.seller_price is not None:
            if self.state_seller2.seller_price <= self.state_seller2.buyer_price:
                return self.state_seller2.seller_price
            price_diff = abs(self.state_seller2.buyer_price - self.state_seller2.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
        return self.state_seller2.seller_price
    
    def _calculate_reward(self) -> float:
        """Calculate global reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a seller, use that seller's min_price for calculation.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected seller's min_price
            selected_seller_min_price = None
            if self.selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if selected_seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller{self.selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        
        If deal is reached:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached with this seller:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another seller:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this seller
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            # Deal reached with this seller: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Get this seller's min_price
            seller_min_price = None
            if seller_id == 1:
                seller_min_price = self.seller1_min_price
            elif seller_id == 2:
                seller_min_price = self.seller2_min_price
            
            # Calculate seller profit: deal_price - seller_min_price
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller{seller_id}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another seller: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this seller)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offers in this round with all sellers.
        This is calculated every round, not just at the end.
        Buyer's reward is aggregated across all sellers using the specified aggregation method.
        
        reward = aggregated(buyer savings with each seller) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer rewards with each seller
        buyer_rewards = []
        
        # Buyer reward with seller1
        if self.state_seller1.buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings_s1 = self.buyer_max_price - self.state_seller1.buyer_price
            reward_s1 = buyer_savings_s1 * self.reward_weights["buyer_savings"]
            buyer_rewards.append(reward_s1)
        
        # Buyer reward with seller2
        if self.state_seller2.buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings_s2 = self.buyer_max_price - self.state_seller2.buyer_price
            reward_s2 = buyer_savings_s2 * self.reward_weights["buyer_savings"]
            buyer_rewards.append(reward_s2)
        
        # Aggregate buyer rewards
        if buyer_rewards:
            aggregated_reward = self._aggregate_rewards(buyer_rewards, self.buyer_reward_aggregation)
        else:
            aggregated_reward = 0.0
        
        # Add round cost (negative penalty)
        reward = aggregated_reward + round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller state
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_state = self.state_seller1
            seller_min_price = self.seller1_min_price
        elif seller_id == 2:
            seller_state = self.state_seller2
            seller_min_price = self.seller2_min_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _aggregate_rewards(self, rewards: list, method: str) -> float:
        """Aggregate multiple rewards using specified method
        
        Args:
            rewards: List of reward values to aggregate
            method: Aggregation method - "average", "max", or "min"
        
        Returns:
            Aggregated reward value
        """
        if not rewards:
            return 0.0
        
        if method == "average":
            return sum(rewards) / len(rewards)
        elif method == "max":
            return max(rewards)
        elif method == "min":
            return min(rewards)
        else:
            # Default to average if unknown method
            return sum(rewards) / len(rewards)
    
    def _get_selected_seller_min_price(self) -> Optional[float]:
        """Get the selected seller's min_price
        
        Returns:
            Selected seller's min_price, or None if no seller is selected
        """
        if self.selected_seller == 1:
            return self.seller1_min_price
        elif self.selected_seller == 2:
            return self.seller2_min_price
        return None
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Uses the selected seller's min_price for calculation.
        If no seller is selected, calculates failure penalty.
        
        Let:
        - buyer_max_price = maximum price the buyer is willing to pay
        - seller_min_price = minimum price the selected seller is willing to accept
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
            D = deal_score_weight (default: 10)
            W = quality_score_weight (default: 80)
            E = efficiency_score_weight (default: 10)
            F = failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
            T = max_rounds
        
        Returns:
            GlobalScore value (only calculated at final result)
        """
        # Get selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Calculate Z
        Z = self.buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (self.buyer_max_price - final_price) / Z
            u_s = (final_price - selected_seller_min_price) / Z
            
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  u_s = (final_price({final_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore based on the formula
        
        u_b = (buyer_max_price - p) / (buyer_max_price - seller_min_price)
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            BuyerScore = discount * (Db + Wb * u_b + Eb)
        Else:
            BuyerScore = -Fb * (1 - discount)
        
        Settings:
            Db = buyer_deal_weight (default: 10)
            Wb = buyer_utility_weight (default: 80)
            Eb = buyer_efficiency_weight (default: 10)
            Fb = buyer_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            BuyerScore value (only calculated at final result)
        """
        # Get selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Calculate Z
        Z = self.buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (self.buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  BuyerScore = discount({discount:.6f}) * (Db({self.buyer_deal_weight:.1f}) + Wb({self.buyer_utility_weight:.1f}) * u_b({u_b:.4f}) + Eb({self.buyer_efficiency_weight:.1f}))")
                print(f"  BuyerScore = {discount:.6f} * ({self.buyer_deal_weight:.1f} + {self.buyer_utility_weight * u_b:.4f} + {self.buyer_efficiency_weight:.1f}) = {buyer_score:.3f}")
            
            return buyer_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
        Uses the selected seller's min_price for calculation.
        If no seller is selected or deal not reached with this seller, calculates failure penalty.
        
        u_s = (p - seller_min_price) / (buyer_max_price - seller_min_price)
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            SellerScore = discount * (Ds + Ws * u_s + Es)
        Else:
            SellerScore = -Fs * (1 - discount)
        
        Settings:
            Ds = seller_deal_weight (default: 10)
            Ws = seller_utility_weight (default: 80)
            Es = seller_efficiency_weight (default: 10)
            Fs = seller_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            SellerScore value (only calculated at final result)
        """
        # Get selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Calculate Z
        Z = self.buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - selected_seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_s = (final_price({final_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f}) * u_s({u_s:.4f}) + Es({self.seller_efficiency_weight:.1f}))")
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            
            return seller_score
    
    def _print_global_score_details(self):
        """Print GlobalScore calculation details (called from example code after Step Rewards)"""
        self._calculate_global_score(print_details=True)
    
    def _print_buyer_score_details(self):
        """Print BuyerScore calculation details (called from example code after Step Rewards)"""
        self._calculate_buyer_score(print_details=True)
    
    def _print_seller_score_details(self):
        """Print SellerScore calculation details (called from example code after Step Rewards)"""
        self._calculate_seller_score(print_details=True)

