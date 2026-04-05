"""Negotiation Environment Implementation"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task1BasicPriceNegotiation(BaseEnv):
    """Negotiation Environment
    
    Manages multi-round negotiation process between buyer and seller.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
        gamma: float = 0.99,
        deal_score_weight: float = 10.0,  # D: deal score weight (default: 30.0 -> 10.0)
        quality_score_weight: float = 80.0,  # W: quality score weight (default: 55.0 -> 80.0)
        efficiency_score_weight: float = 10.0,  # E: efficiency score weight (default: 15.0 -> 10.0)
        failure_penalty_weight: float = 15.0,
        buyer_deal_weight: float = 10.0,  # Db: buyer deal weight (default: 30.0 -> 10.0)
        buyer_utility_weight: float = 80.0,  # Wb: buyer utility weight (default: 55.0 -> 80.0)
        buyer_efficiency_weight: float = 10.0,  # Eb: buyer efficiency weight (default: 15.0 -> 10.0)
        buyer_failure_penalty_weight: float = 15.0,
        seller_deal_weight: float = 10.0,  # Ds: seller deal weight (default: 30.0 -> 10.0)
        seller_utility_weight: float = 80.0,  # Ws: seller utility weight (default: 55.0 -> 80.0)
        seller_efficiency_weight: float = 10.0,  # Es: seller efficiency weight (default: 15.0 -> 10.0)
        seller_failure_penalty_weight: float = 15.0,
    ):
        """Initialize negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial price offered by seller
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller_min_price: Minimum acceptable price for seller (confidential)
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
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer_max_price = buyer_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
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
        
        # Set default reward weights
        default_weights = {
            "buyer_savings": 1.0,      # Buyer savings weight
            "seller_profit": 1.0,      # Seller profit weight
            "time_cost": 0.1,          # Time cost weight (reduced impact)
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
        # Call parent class initialization
        super().__init__()
        
        # State management
        self.memory = ConversationMemory()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description
            product_info: Product information
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory.clear()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Extract product_images for VLM (from kwargs, product_info, or product_info sub-fields)
        product_info = product_info or {}
        product_images = kwargs.get("product_images")
        if product_images is None:
            product_images = product_info.get("product_images") or product_info.get("images")
        if product_images is None:
            # Support single image in product_info
            img_path = product_info.get("image_path") or product_info.get("image_url")
            if img_path is not None:
                product_images = [img_path]
        if product_images is not None and not isinstance(product_images, list):
            product_images = [product_images]
        self.product_images = product_images  # Store for observation
        
        # Initialize Agents
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info,  # Buyer can now see product information
            "product_images": product_images,  # For VLM: product images (path/URL)
        }
        self.buyer_agent.initialize(buyer_context)
        
        seller_context = {
            "product_info": product_info,
            "initial_price": self.initial_seller_price,
            "min_price": self.seller_min_price,
            "environment_info": self.environment_info,
            "product_images": product_images,  # For VLM: product images (path/URL)
        }
        self.seller_agent.initialize(seller_context)
        
        # No initial seller offer - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer_action: Optional[str] = None, 
        seller_action: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, both buyer and seller make their responses, then check if agreement is reached.
        
        Args:
            buyer_action: Buyer's response text (optional, can be None if buyer doesn't respond)
            seller_action: Seller's response text (optional, can be None if seller doesn't respond)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory (both agents respond in the same round)
        # Buyer responds first, then seller
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            # Extract buyer price
            buyer_price = self._extract_price(buyer_action)
            if buyer_price is not None:
                self.state.update(buyer_price=buyer_price)
                self.negotiation_info.buyer_price = buyer_price
                self.negotiation_info.current_price = buyer_price
        
        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            # Extract seller price
            seller_price = self._extract_price(seller_action)
            if seller_price is not None:
                self.state.update(seller_price=seller_price)
                self.negotiation_info.seller_price = seller_price
                self.negotiation_info.current_price = seller_price
        
        # Check if agreement is reached (after both agents have responded)
        terminated = False
        truncated = False
        reward = 0.0
        seller_reward = 0.0
        buyer_reward = 0.0
        
        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # When seller's offer <= buyer's offer: deal at seller's price; otherwise use midpoint
            if self.state.seller_price <= self.state.buyer_price:
                agreed_price = self.state.seller_price
            else:
                agreed_price = (self.state.buyer_price + self.state.seller_price) / 2
            self.state.update(agreed_price=agreed_price)
            self.negotiation_info.current_price = agreed_price
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        step_seller_reward = self._calculate_step_seller_reward()
        step_buyer_reward = self._calculate_step_buyer_reward()
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_seller_reward"] = step_seller_reward
        info["step_buyer_reward"] = step_buyer_reward
        
        # Calculate final scores only after the last round is completed
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["seller_reward"] = seller_reward
            info["buyer_reward"] = buyer_reward
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
        history = self.memory.get_history()
        if history:
            # Determine which round's messages to display
            # Messages are stored with the round value at the time of storage (before current_round is incremented)
            # In step(), messages are added first, then current_round is incremented
            # So for any completed round, messages are stored at current_round - 1
            # This applies to both AGREED/TIMEOUT (where round was incremented after agreement check)
            # and ONGOING (where round was incremented for next round)
            round_to_display = self.current_round - 1 if self.current_round > 0 else 0
            
            round_messages = [
                msg for msg in history if msg["round"] == round_to_display
            ]
            
            if round_messages:
                # Display round number: current_round is already incremented, so it represents the completed round number
                display_round = self.current_round
                output_lines.append(f"\n{'='*60}")
                output_lines.append(f"Round {display_round} - Negotiation Output")
                output_lines.append(f"{'='*60}")
                
                # Display buyer message first (if exists)
                buyer_msg = next(
                    (msg for msg in round_messages if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg:
                    output_lines.append(f"\n[BUYER Output]:")
                    output_lines.append(f"  {buyer_msg['content']}")
                
                # Display seller message (if exists)
                seller_msg = next(
                    (msg for msg in round_messages if msg["role"] == "seller"), 
                    None
                )
                if seller_msg:
                    output_lines.append(f"\n[SELLER Output]:")
                    output_lines.append(f"  {seller_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display prices
        # If agreement reached, show agreed price for both; otherwise show individual prices
        if self.state.agreed_price is not None:
            # Agreement reached: show agreed price for both buyer and seller
            output_lines.append(f"  Buyer Price: ${self.state.agreed_price:.2f}")
            output_lines.append(f"  Seller Price: ${self.state.agreed_price:.2f}")
        else:
            # No agreement yet: show individual prices
            if self.state.buyer_price is not None:
                output_lines.append(f"  Buyer Price: ${self.state.buyer_price:.2f}")
            else:
                output_lines.append(f"  Buyer Price: Not specified")
            
            if self.state.seller_price is not None:
                output_lines.append(f"  Seller Price: ${self.state.seller_price:.2f}")
            else:
                output_lines.append(f"  Seller Price: Not specified")
        
        # Check agreement status and provide reason
        is_agreed = self._check_agreement()
        agreement_reason = self._get_agreement_reason()
        
        output_lines.append(f"  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        output_lines.append(f"  Reason: {agreement_reason}")
        
        # Display agreed price if agreement is reached
        if self.state.agreed_price is not None:
            output_lines.append(f"  Agreed Price: ${self.state.agreed_price:.2f}")
        
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
    
    def _get_agreement_reason(self) -> str:
        """Get reason for agreement or non-agreement
        
        Returns:
            String describing why agreement was or was not reached
        """
        if self.state.buyer_price is None or self.state.seller_price is None:
            if self.state.buyer_price is None and self.state.seller_price is None:
                return "Both buyer and seller prices are not specified yet"
            elif self.state.buyer_price is None:
                return "Buyer price is not specified yet"
            else:
                return "Seller price is not specified yet"
        
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        
        if self.state.seller_price <= self.state.buyer_price:
            return f"Seller's offer (${self.state.seller_price:.2f}) <= buyer's offer (${self.state.buyer_price:.2f}), deal at seller's price"
        if price_diff <= self.price_tolerance:
            return f"Price difference (${price_diff:.2f}) is within tolerance (${self.price_tolerance:.2f})"
        else:
            return f"Price difference (${price_diff:.2f}) exceeds tolerance (${self.price_tolerance:.2f})"
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory.clear()
        self.state = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "seller_price": self.state.seller_price,
            "buyer_price": self.state.buyer_price,
            "status": self.negotiation_info.status.value,
        }
        # Include product_images for VLM agents
        if getattr(self, "product_images", None):
            obs["product_images"] = self.product_images
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller_price": self.state.seller_price,
            "buyer_price": self.state.buyer_price,
            "agreed_price": self.state.agreed_price,
            "negotiation_info": self.negotiation_info,
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
    
    def _check_agreement(self) -> bool:
        """Check if agreement is reached
        
        Agreement is reached when:
        1. The prices of both buyer and seller are within the tolerance range, or
        2. Seller's offer is less than or equal to buyer's offer (seller_price <= buyer_price).
        
        Returns:
            Whether agreement is reached
        """
        if self.state.buyer_price is None or self.state.seller_price is None:
            return False
        
        # Condition 1: within tolerance
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        if price_diff <= self.price_tolerance:
            return True
        # Condition 2: seller's offer <= buyer's offer -> deal reached
        if self.state.seller_price <= self.state.buyer_price:
            return True
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: buyer savings + seller profit + time cost
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: seller profit + time cost
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Seller Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: buyer savings + time cost
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Buyer Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
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
    
    def _calculate_step_seller_reward(self) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if self.state.seller_price is not None and self.seller_min_price is not None:
            seller_profit = self.state.seller_price - self.seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_buyer_reward(self) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If buyer_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        buyer_savings = 0.0
        
        # Calculate buyer savings from current offer: buyer_max_price - buyer_price
        if self.state.buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings = self.buyer_max_price - self.state.buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Let:
        - buyer_max_price = maximum price the buyer is willing to pay
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
        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1)
        # Note: Formula uses t-1 where t is 1-based (first round = 1)
        # current_round = completed round count, so round_index = current_round - 1
        # This means: 1 round completed -> γ^0=1, 2 rounds completed -> γ^1, etc.
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price (agreed_price if available, otherwise use current prices)
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            # Use average if both prices are available but not agreed
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (self.buyer_max_price - final_price) / Z
            u_s = (final_price - self.seller_min_price) / Z
            
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({self.seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
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
            Db = buyer_deal_weight (default: 30)
            Wb = buyer_utility_weight (default: 55)
            Eb = buyer_efficiency_weight (default: 15)
            Fb = buyer_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            BuyerScore value (only calculated at final result)
        """
        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1) where t is completed round count (1-based)
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (self.buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
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
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
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
        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1) where t is completed round count (1-based)
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - self.seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({self.seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f}) * u_s({u_s:.4f}) + Es({self.seller_efficiency_weight:.1f}))")
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            
            return seller_score
    
    def _print_global_score_details(self):
        """Print GlobalScore calculation details (called from render() after Round Summary)"""
        self._calculate_global_score(print_details=True)
    
    def _print_buyer_score_details(self):
        """Print BuyerScore calculation details (called from render() after Round Summary)"""
        self._calculate_buyer_score(print_details=True)
    
    def _print_seller_score_details(self):
        """Print SellerScore calculation details (called from render() after Round Summary)"""
        self._calculate_seller_score(print_details=True)

