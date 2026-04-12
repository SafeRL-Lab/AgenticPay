"""Task2 Parallel Three-Buyer Three-Seller Two-Product Negotiation Environment Implementation

Supports parallel negotiation between three buyers and three sellers for two products.
Each buyer can negotiate with each seller, and deals are matched based on mutual agreement.
Prices represent total price for both products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation(BaseEnv):
    """Task2 Parallel Three-Buyer Three-Seller Two-Product Negotiation Environment
    
    Manages parallel negotiation process between three buyers and three sellers for two products.
    Each buyer negotiates with all three sellers simultaneously, and each seller negotiates with all three buyers.
    Deals are matched when both buyer and seller agree on a price within tolerance.
    Prices represent total price for both products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        buyer3_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        seller3_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 200.0,
        initial_seller2_price: float = 220.0,
        initial_seller3_price: float = 240.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        buyer3_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        seller3_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
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
        """Initialize multi-buyer multi-seller multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            buyer3_agent: Third Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            seller3_agent: Third Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial total price offered by seller1 for both products
            initial_seller2_price: Initial total price offered by seller2 for both products
            initial_seller3_price: Initial total price offered by seller3 for both products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for both products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for both products)
            buyer3_max_price: Maximum acceptable total price for buyer3 (confidential, for both products)
            seller1_min_price: Minimum acceptable total price for seller1 (confidential, for both products)
            seller2_min_price: Minimum acceptable total price for seller2 (confidential, for both products)
            seller3_min_price: Minimum acceptable total price for seller3 (confidential, for both products)
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
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.buyer3_agent = buyer3_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.seller3_agent = seller3_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.initial_seller3_price = initial_seller3_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.buyer3_max_price = buyer3_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
        self.seller3_min_price = seller3_min_price
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
        
        # State management - separate for each buyer-seller pair (3x3 = 9 pairs)
        # buyer1-seller1, buyer1-seller2, buyer1-seller3
        self.memory_b1s1 = ConversationMemory()
        self.state_b1s1 = NegotiationState()
        self.memory_b1s2 = ConversationMemory()
        self.state_b1s2 = NegotiationState()
        self.memory_b1s3 = ConversationMemory()
        self.state_b1s3 = NegotiationState()
        # buyer2-seller1, buyer2-seller2, buyer2-seller3
        self.memory_b2s1 = ConversationMemory()
        self.state_b2s1 = NegotiationState()
        self.memory_b2s2 = ConversationMemory()
        self.state_b2s2 = NegotiationState()
        self.memory_b2s3 = ConversationMemory()
        self.state_b2s3 = NegotiationState()
        # buyer3-seller1, buyer3-seller2, buyer3-seller3
        self.memory_b3s1 = ConversationMemory()
        self.state_b3s1 = NegotiationState()
        self.memory_b3s2 = ConversationMemory()
        self.state_b3s2 = NegotiationState()
        self.memory_b3s3 = ConversationMemory()
        self.state_b3s3 = NegotiationState()
        
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which buyer-seller pair made the deal
        self.selected_buyer: Optional[int] = None  # 1, 2, or 3
        self.selected_seller: Optional[int] = None  # 1, 2, or 3
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
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b1s3.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.memory_b2s3.clear()
        self.memory_b3s1.clear()
        self.memory_b3s2.clear()
        self.memory_b3s3.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b1s3 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.state_b2s3 = NegotiationState()
        self.state_b3s1 = NegotiationState()
        self.state_b3s2 = NegotiationState()
        self.state_b3s3 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.selected_buyer = None
        self.selected_seller = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Initialize Buyer1 Agent (buyer1 knows about all three sellers and both products)
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 1,
            "num_sellers": 3,
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent (buyer2 knows about all three sellers and both products)
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 2,
            "num_sellers": 3,
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Buyer3 Agent (buyer3 knows about all three sellers and both products)
        buyer3_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer3_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 3,
            "num_sellers": 3,
        }
        self.buyer3_agent.initialize(buyer3_context)
        
        # Initialize Seller1 Agent (seller1 knows about all three buyers and both products)
        seller1_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller1_price,  # Initial total price
            "min_price": self.seller1_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 1,
            "num_buyers": 3,
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent (seller2 knows about all three buyers and both products)
        seller2_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller2_price,  # Initial total price
            "min_price": self.seller2_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 2,
            "num_buyers": 3,
        }
        self.seller2_agent.initialize(seller2_context)
        
        # Initialize Seller3 Agent (seller3 knows about all three buyers and both products)
        seller3_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller3_price,  # Initial total price
            "min_price": self.seller3_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 3,
            "num_buyers": 3,
        }
        self.seller3_agent.initialize(seller3_context)
        
        # No initial seller offer - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        buyer1_action_seller1: Optional[str] = None,
        buyer1_action_seller2: Optional[str] = None,
        buyer1_action_seller3: Optional[str] = None,
        buyer2_action_seller1: Optional[str] = None,
        buyer2_action_seller2: Optional[str] = None,
        buyer2_action_seller3: Optional[str] = None,
        buyer3_action_seller1: Optional[str] = None,
        buyer3_action_seller2: Optional[str] = None,
        buyer3_action_seller3: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller1_action_buyer3: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
        seller2_action_buyer3: Optional[str] = None,
        seller3_action_buyer1: Optional[str] = None,
        seller3_action_buyer2: Optional[str] = None,
        seller3_action_buyer3: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, all buyers respond to all sellers first, then all sellers respond to all buyers.
        Order: buyer -> seller
        Prices represent total price for both products.
        
        Args:
            buyer1_action_seller1: Buyer1's response to seller1 (optional)
            buyer1_action_seller2: Buyer1's response to seller2 (optional)
            buyer1_action_seller3: Buyer1's response to seller3 (optional)
            buyer2_action_seller1: Buyer2's response to seller1 (optional)
            buyer2_action_seller2: Buyer2's response to seller2 (optional)
            buyer2_action_seller3: Buyer2's response to seller3 (optional)
            buyer3_action_seller1: Buyer3's response to seller1 (optional)
            buyer3_action_seller2: Buyer3's response to seller2 (optional)
            buyer3_action_seller3: Buyer3's response to seller3 (optional)
            seller1_action_buyer1: Seller1's response to buyer1 (optional)
            seller1_action_buyer2: Seller1's response to buyer2 (optional)
            seller1_action_buyer3: Seller1's response to buyer3 (optional)
            seller2_action_buyer1: Seller2's response to buyer1 (optional)
            seller2_action_buyer2: Seller2's response to buyer2 (optional)
            seller2_action_buyer3: Seller2's response to buyer3 (optional)
            seller3_action_buyer1: Seller3's response to buyer1 (optional)
            seller3_action_buyer2: Seller3's response to buyer2 (optional)
            seller3_action_buyer3: Seller3's response to buyer3 (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Process buyer actions first
        # buyer1-seller1, buyer1-seller2, buyer1-seller3
        if buyer1_action_seller1 is not None:
            self.memory_b1s1.add_message("buyer", buyer1_action_seller1, self.current_round)
            buyer_price = self._extract_price(buyer1_action_seller1)
            if buyer_price is not None:
                self.state_b1s1.update(buyer_price=buyer_price)
        if buyer1_action_seller2 is not None:
            self.memory_b1s2.add_message("buyer", buyer1_action_seller2, self.current_round)
            buyer_price = self._extract_price(buyer1_action_seller2)
            if buyer_price is not None:
                self.state_b1s2.update(buyer_price=buyer_price)
        if buyer1_action_seller3 is not None:
            self.memory_b1s3.add_message("buyer", buyer1_action_seller3, self.current_round)
            buyer_price = self._extract_price(buyer1_action_seller3)
            if buyer_price is not None:
                self.state_b1s3.update(buyer_price=buyer_price)
        
        # buyer2-seller1, buyer2-seller2, buyer2-seller3
        if buyer2_action_seller1 is not None:
            self.memory_b2s1.add_message("buyer", buyer2_action_seller1, self.current_round)
            buyer_price = self._extract_price(buyer2_action_seller1)
            if buyer_price is not None:
                self.state_b2s1.update(buyer_price=buyer_price)
        if buyer2_action_seller2 is not None:
            self.memory_b2s2.add_message("buyer", buyer2_action_seller2, self.current_round)
            buyer_price = self._extract_price(buyer2_action_seller2)
            if buyer_price is not None:
                self.state_b2s2.update(buyer_price=buyer_price)
        if buyer2_action_seller3 is not None:
            self.memory_b2s3.add_message("buyer", buyer2_action_seller3, self.current_round)
            buyer_price = self._extract_price(buyer2_action_seller3)
            if buyer_price is not None:
                self.state_b2s3.update(buyer_price=buyer_price)
        
        # buyer3-seller1, buyer3-seller2, buyer3-seller3
        if buyer3_action_seller1 is not None:
            self.memory_b3s1.add_message("buyer", buyer3_action_seller1, self.current_round)
            buyer_price = self._extract_price(buyer3_action_seller1)
            if buyer_price is not None:
                self.state_b3s1.update(buyer_price=buyer_price)
        if buyer3_action_seller2 is not None:
            self.memory_b3s2.add_message("buyer", buyer3_action_seller2, self.current_round)
            buyer_price = self._extract_price(buyer3_action_seller2)
            if buyer_price is not None:
                self.state_b3s2.update(buyer_price=buyer_price)
        if buyer3_action_seller3 is not None:
            self.memory_b3s3.add_message("buyer", buyer3_action_seller3, self.current_round)
            buyer_price = self._extract_price(buyer3_action_seller3)
            if buyer_price is not None:
                self.state_b3s3.update(buyer_price=buyer_price)
        
        # Process seller actions after buyers
        # seller1-buyer1, seller1-buyer2, seller1-buyer3
        if seller1_action_buyer1 is not None:
            self.memory_b1s1.add_message("seller", seller1_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer1)
            if seller_price is not None:
                self.state_b1s1.update(seller_price=seller_price)
        if seller1_action_buyer2 is not None:
            self.memory_b2s1.add_message("seller", seller1_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer2)
            if seller_price is not None:
                self.state_b2s1.update(seller_price=seller_price)
        if seller1_action_buyer3 is not None:
            self.memory_b3s1.add_message("seller", seller1_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer3)
            if seller_price is not None:
                self.state_b3s1.update(seller_price=seller_price)
        
        # seller2-buyer1, seller2-buyer2, seller2-buyer3
        if seller2_action_buyer1 is not None:
            self.memory_b1s2.add_message("seller", seller2_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer1)
            if seller_price is not None:
                self.state_b1s2.update(seller_price=seller_price)
        if seller2_action_buyer2 is not None:
            self.memory_b2s2.add_message("seller", seller2_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer2)
            if seller_price is not None:
                self.state_b2s2.update(seller_price=seller_price)
        if seller2_action_buyer3 is not None:
            self.memory_b3s2.add_message("seller", seller2_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer3)
            if seller_price is not None:
                self.state_b3s2.update(seller_price=seller_price)
        
        # seller3-buyer1, seller3-buyer2, seller3-buyer3
        if seller3_action_buyer1 is not None:
            self.memory_b1s3.add_message("seller", seller3_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer1)
            if seller_price is not None:
                self.state_b1s3.update(seller_price=seller_price)
        if seller3_action_buyer2 is not None:
            self.memory_b2s3.add_message("seller", seller3_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer2)
            if seller_price is not None:
                self.state_b2s3.update(seller_price=seller_price)
        if seller3_action_buyer3 is not None:
            self.memory_b3s3.add_message("seller", seller3_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer3)
            if seller_price is not None:
                self.state_b3s3.update(seller_price=seller_price)
        
        # Check for deals: each buyer-seller pair can make a deal (3x3 = 9 pairs)
        # A deal is made when both buyer and seller want to make a deal and prices are within tolerance
        deals = []  # List of (buyer_id, seller_id, price) tuples
        
        # Check all 9 buyer-seller pairs
        buyer_seller_pairs = [
            (1, 1, buyer1_action_seller1, seller1_action_buyer1, self.state_b1s1),
            (1, 2, buyer1_action_seller2, seller2_action_buyer1, self.state_b1s2),
            (1, 3, buyer1_action_seller3, seller3_action_buyer1, self.state_b1s3),
            (2, 1, buyer2_action_seller1, seller1_action_buyer2, self.state_b2s1),
            (2, 2, buyer2_action_seller2, seller2_action_buyer2, self.state_b2s2),
            (2, 3, buyer2_action_seller3, seller3_action_buyer2, self.state_b2s3),
            (3, 1, buyer3_action_seller1, seller1_action_buyer3, self.state_b3s1),
            (3, 2, buyer3_action_seller2, seller2_action_buyer3, self.state_b3s2),
            (3, 3, buyer3_action_seller3, seller3_action_buyer3, self.state_b3s3),
        ]
        
        for buyer_id, seller_id, buyer_action, seller_action, state in buyer_seller_pairs:
            if (buyer_action is not None and seller_action is not None and
                state.buyer_price is not None and
                state.seller_price is not None):
                price_diff = abs(state.buyer_price - state.seller_price)
                if price_diff <= self.price_tolerance:
                    deal_price = (state.buyer_price + state.seller_price) / 2
                    deals.append((buyer_id, seller_id, deal_price))
                elif state.seller_price <= state.buyer_price:
                    deal_price = state.seller_price
                    deals.append((buyer_id, seller_id, deal_price))
        
        # Select the best deal: prioritize buyer's preference (lower price) and seller's preference (higher price)
        # If multiple deals exist, choose the one with the best price for both parties
        if deals:
            # For buyers: prefer lower price, for sellers: prefer higher price
            # We'll choose the deal that maximizes overall utility (buyer savings + seller profit)
            best_deal = None
            best_utility = float('-inf')
            
            for buyer_id, seller_id, price in deals:
                buyer_max = (self.buyer1_max_price if buyer_id == 1 else
                            self.buyer2_max_price if buyer_id == 2 else
                            self.buyer3_max_price)
                seller_min = (self.seller1_min_price if seller_id == 1 else
                             self.seller2_min_price if seller_id == 2 else
                             self.seller3_min_price)
                
                buyer_savings = (buyer_max - price) if buyer_max is not None else 0
                seller_profit = (price - seller_min) if seller_min is not None else 0
                utility = buyer_savings + seller_profit
                
                if utility > best_utility:
                    best_utility = utility
                    best_deal = (buyer_id, seller_id, price)
            
            if best_deal:
                self.selected_buyer, self.selected_seller, self.final_deal_price = best_deal
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        buyer3_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        seller3_reward = 0.0
        
        if self.selected_buyer is not None and self.selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            buyer3_reward = self._calculate_buyer_reward(3)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            seller3_reward = self._calculate_seller_reward(3)
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
            buyer3_reward = self._calculate_buyer_reward(3)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            seller3_reward = self._calculate_seller_reward(3)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        step_buyer1_reward = self._calculate_step_buyer_reward(1)
        step_buyer2_reward = self._calculate_step_buyer_reward(2)
        step_buyer3_reward = self._calculate_step_buyer_reward(3)
        step_seller1_reward = self._calculate_step_seller_reward(1)
        step_seller2_reward = self._calculate_step_seller_reward(2)
        step_seller3_reward = self._calculate_step_seller_reward(3)
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer1_reward"] = step_buyer1_reward
        info["step_buyer2_reward"] = step_buyer2_reward
        info["step_buyer3_reward"] = step_buyer3_reward
        info["step_seller1_reward"] = step_seller1_reward
        info["step_seller2_reward"] = step_seller2_reward
        info["step_seller3_reward"] = step_seller3_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.selected_buyer
                info["selected_seller"] = self.selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["buyer3_reward"] = buyer3_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            info["seller3_reward"] = seller3_reward
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
        
        Displays all buyer-seller conversations for each round, followed by a round summary.
        Prices shown are total prices for both products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
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
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history_b1s1 = self.memory_b1s1.get_history()
        history_b1s2 = self.memory_b1s2.get_history()
        history_b1s3 = self.memory_b1s3.get_history()
        history_b2s1 = self.memory_b2s1.get_history()
        history_b2s2 = self.memory_b2s2.get_history()
        history_b2s3 = self.memory_b2s3.get_history()
        history_b3s1 = self.memory_b3s1.get_history()
        history_b3s2 = self.memory_b3s2.get_history()
        history_b3s3 = self.memory_b3s3.get_history()
        
        # Determine which round's messages to display
        # Messages are stored with the round value at the time of storage (before current_round is incremented)
        # In step(), messages are added first, then current_round is incremented
        # So for any completed round, messages are stored at current_round - 1
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        
        # Determine display round number
        if self.negotiation_info.status in [NegotiationStatus.AGREED, NegotiationStatus.TIMEOUT]:
            display_round = self.current_round
        else:
            display_round = self.current_round if self.current_round > 0 else 0
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Parallel Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display all 9 buyer-seller conversations
        conversations = [
            ("BUYER 1 - SELLER 1", history_b1s1),
            ("BUYER 1 - SELLER 2", history_b1s2),
            ("BUYER 1 - SELLER 3", history_b1s3),
            ("BUYER 2 - SELLER 1", history_b2s1),
            ("BUYER 2 - SELLER 2", history_b2s2),
            ("BUYER 2 - SELLER 3", history_b2s3),
            ("BUYER 3 - SELLER 1", history_b3s1),
            ("BUYER 3 - SELLER 2", history_b3s2),
            ("BUYER 3 - SELLER 3", history_b3s3),
        ]
        
        for title, history in conversations:
            output_lines.append(f"\n[{title} Conversation]:")
            if history:
                round_messages = [
                    msg for msg in history if msg["round"] == round_to_display
                ]
                if round_messages:
                    # Display buyer message first (if exists)
                    buyer_msg = next(
                        (msg for msg in round_messages if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg:
                        output_lines.append(f"  [BUYER]: {buyer_msg['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg = next(
                        (msg for msg in round_messages if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg:
                        output_lines.append(f"  [SELLER]: {seller_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display prices for each pair (total for both products)
        price_pairs = [
            ("Buyer1-Seller1", self.state_b1s1),
            ("Buyer1-Seller2", self.state_b1s2),
            ("Buyer1-Seller3", self.state_b1s3),
            ("Buyer2-Seller1", self.state_b2s1),
            ("Buyer2-Seller2", self.state_b2s2),
            ("Buyer2-Seller3", self.state_b2s3),
            ("Buyer3-Seller1", self.state_b3s1),
            ("Buyer3-Seller2", self.state_b3s2),
            ("Buyer3-Seller3", self.state_b3s3),
        ]
        
        for pair_name, state in price_pairs:
            output_lines.append(f"\n{pair_name}:")
            if state.buyer_price is not None:
                output_lines.append(f"  Buyer Total Price: ${state.buyer_price:.2f}")
            else:
                output_lines.append(f"  Buyer Total Price: Not specified")
            if state.seller_price is not None:
                output_lines.append(f"  Seller Total Price: ${state.seller_price:.2f}")
            else:
                output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.selected_buyer is not None and self.selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE: Buyer {self.selected_buyer} - Seller {self.selected_seller}")
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
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b1s3.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.memory_b2s3.clear()
        self.memory_b3s1.clear()
        self.memory_b3s2.clear()
        self.memory_b3s3.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b1s3 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.state_b2s3 = NegotiationState()
        self.state_b3s1 = NegotiationState()
        self.state_b3s2 = NegotiationState()
        self.state_b3s3 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_b1s1": self.memory_b1s1.get_history(),
            "conversation_history_b1s2": self.memory_b1s2.get_history(),
            "conversation_history_b1s3": self.memory_b1s3.get_history(),
            "conversation_history_b2s1": self.memory_b2s1.get_history(),
            "conversation_history_b2s2": self.memory_b2s2.get_history(),
            "conversation_history_b2s3": self.memory_b2s3.get_history(),
            "conversation_history_b3s1": self.memory_b3s1.get_history(),
            "conversation_history_b3s2": self.memory_b3s2.get_history(),
            "conversation_history_b3s3": self.memory_b3s3.get_history(),
            "current_round": self.current_round,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b1s3_buyer_price": self.state_b1s3.buyer_price,  # Total price for both products
            "b1s3_seller_price": self.state_b1s3.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "b2s3_buyer_price": self.state_b2s3.buyer_price,  # Total price for both products
            "b2s3_seller_price": self.state_b2s3.seller_price,  # Total price for both products
            "b3s1_buyer_price": self.state_b3s1.buyer_price,  # Total price for both products
            "b3s1_seller_price": self.state_b3s1.seller_price,  # Total price for both products
            "b3s2_buyer_price": self.state_b3s2.buyer_price,  # Total price for both products
            "b3s2_seller_price": self.state_b3s2.seller_price,  # Total price for both products
            "b3s3_buyer_price": self.state_b3s3.buyer_price,  # Total price for both products
            "b3s3_seller_price": self.state_b3s3.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "product_info": self.product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b1s3_buyer_price": self.state_b1s3.buyer_price,  # Total price for both products
            "b1s3_seller_price": self.state_b1s3.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "b2s3_buyer_price": self.state_b2s3.buyer_price,  # Total price for both products
            "b2s3_seller_price": self.state_b2s3.seller_price,  # Total price for both products
            "b3s1_buyer_price": self.state_b3s1.buyer_price,  # Total price for both products
            "b3s1_seller_price": self.state_b3s1.seller_price,  # Total price for both products
            "b3s2_buyer_price": self.state_b3s2.buyer_price,  # Total price for both products
            "b3s2_seller_price": self.state_b3s2.seller_price,  # Total price for both products
            "b3s3_buyer_price": self.state_b3s3.buyer_price,  # Total price for both products
            "b3s3_seller_price": self.state_b3s3.seller_price,  # Total price for both products
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
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
        """Check if agent wants to make a deal
        
        Args:
            text: Agent's response text
            
        Returns:
            Whether agent wants to make a deal
        """
        if 'MAKE_DEAL' in text.upper():
            return True
        
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
    
    def _calculate_reward(self) -> float:
        """Calculate global reward
        
        Returns:
            Reward value
        """
        time_cost = -self.current_round
        
        if (self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_buyer is not None and
            self.selected_seller is not None and
            self.final_deal_price is not None):
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            buyer_max_price = (self.buyer1_max_price if self.selected_buyer == 1 else
                              self.buyer2_max_price if self.selected_buyer == 2 else
                              self.buyer3_max_price)
            seller_min_price = (self.seller1_min_price if self.selected_seller == 1 else
                               self.seller2_min_price if self.selected_seller == 2 else
                               self.seller3_min_price)
            
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Global Reward = buyer{self.selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.selected_buyer}_max={buyer_max_price}, deal_price={deal_price:.2f}, seller{self.selected_seller}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective
        
        Args:
            buyer_id: Buyer ID (1, 2, or 3)
        
        Returns:
            Reward value from buyer's perspective
        """
        time_cost = -self.current_round
        
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            buyer_max_price = (self.buyer1_max_price if buyer_id == 1 else
                              self.buyer2_max_price if buyer_id == 2 else
                              self.buyer3_max_price)
            
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Buyer{buyer_id} Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{buyer_id}_max={buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this buyer)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Args:
            seller_id: Seller ID (1, 2, or 3)
        
        Returns:
            Reward value from seller's perspective
        """
        time_cost = -self.current_round
        
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            seller_min_price = (self.seller1_min_price if seller_id == 1 else
                               self.seller2_min_price if seller_id == 2 else
                               self.seller3_min_price)
            
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Seller{seller_id} Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller{seller_id}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this seller)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Args:
            buyer_id: Buyer ID (1, 2, or 3)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        round_cost = -self.current_round
        
        buyer_rewards = []
        buyer_max_price = (self.buyer1_max_price if buyer_id == 1 else
                          self.buyer2_max_price if buyer_id == 2 else
                          self.buyer3_max_price)
        
        # Buyer reward with seller1, seller2, seller3
        states = [
            self.state_b1s1 if buyer_id == 1 else (self.state_b2s1 if buyer_id == 2 else self.state_b3s1),
            self.state_b1s2 if buyer_id == 1 else (self.state_b2s2 if buyer_id == 2 else self.state_b3s2),
            self.state_b1s3 if buyer_id == 1 else (self.state_b2s3 if buyer_id == 2 else self.state_b3s3),
        ]
        
        for state in states:
            if state.buyer_price is not None and buyer_max_price is not None:
                buyer_savings = buyer_max_price - state.buyer_price
                reward = buyer_savings * self.reward_weights["buyer_savings"]
                buyer_rewards.append(reward)
        
        if buyer_rewards:
            aggregated_reward = self._aggregate_rewards(buyer_rewards, self.buyer_reward_aggregation)
        else:
            aggregated_reward = 0.0
        
        reward = aggregated_reward + round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Args:
            seller_id: Seller ID (1, 2, or 3)
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        round_cost = -self.current_round
        
        seller_rewards = []
        seller_min_price = (self.seller1_min_price if seller_id == 1 else
                           self.seller2_min_price if seller_id == 2 else
                           self.seller3_min_price)
        
        # Seller reward with buyer1, buyer2, buyer3
        states = [
            self.state_b1s1 if seller_id == 1 else (self.state_b1s2 if seller_id == 2 else self.state_b1s3),
            self.state_b2s1 if seller_id == 1 else (self.state_b2s2 if seller_id == 2 else self.state_b2s3),
            self.state_b3s1 if seller_id == 1 else (self.state_b3s2 if seller_id == 2 else self.state_b3s3),
        ]
        
        for state in states:
            if state.seller_price is not None and seller_min_price is not None:
                seller_profit = state.seller_price - seller_min_price
                reward = seller_profit * self.reward_weights["seller_profit"]
                seller_rewards.append(reward)
        
        if seller_rewards:
            aggregated_reward = self._aggregate_rewards(seller_rewards, self.seller_reward_aggregation)
        else:
            aggregated_reward = 0.0
        
        reward = aggregated_reward + round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _get_selected_buyer_max_price(self) -> Optional[float]:
        """Get the final selected buyer's max_price
        
        Returns:
            Final selected buyer's max_price, or None if no buyer is selected
        """
        if self.selected_buyer == 1:
            return self.buyer1_max_price
        elif self.selected_buyer == 2:
            return self.buyer2_max_price
        elif self.selected_buyer == 3:
            return self.buyer3_max_price
        return None
    
    def _get_selected_seller_min_price(self) -> Optional[float]:
        """Get the final selected seller's min_price
        
        Returns:
            Final selected seller's min_price, or None if no seller is selected
        """
        if self.selected_seller == 1:
            return self.seller1_min_price
        elif self.selected_seller == 2:
            return self.seller2_min_price
        elif self.selected_seller == 3:
            return self.seller3_min_price
        return None
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Uses the final selected buyer's max_price and selected seller's min_price for calculation.
        If no buyer or seller is selected, calculates failure penalty.
        
        Let:
        - buyer_max_price = maximum price the final selected buyer is willing to pay
        - seller_min_price = minimum price the final selected seller is willing to accept
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
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if selected_buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  selected_buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Calculate Z
        Z = selected_buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (selected_buyer_max_price - final_price) / Z
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (selected_buyer_max_price({selected_buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore based on the formula
        
        Uses the final selected buyer's max_price and selected seller's min_price for calculation.
        
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
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if selected_buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  selected_buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Calculate Z
        Z = selected_buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (selected_buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
        Uses the final selected buyer's max_price and selected seller's min_price for calculation.
        If no buyer or seller is selected or deal not reached, calculates failure penalty.
        
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
        # Get final selected buyer's max_price and seller's min_price
        selected_buyer_max_price = self._get_selected_buyer_max_price()
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if selected_buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  selected_buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Calculate Z
        Z = selected_buyer_max_price - selected_seller_min_price
        
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= selected_buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - selected_seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
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
                print(f"  Z = selected_buyer_max_price({selected_buyer_max_price:.2f}) - selected_seller_min_price({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= selected_buyer_max_price({selected_buyer_max_price:.2f})) = {valid_range}")
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
            return sum(rewards) / len(rewards)

