import gymnasium as gym
import numpy as np
from energy_net.env.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

def main():
    # Create the environment with required parameters
    env = EnergyNetV0(
        controller_name="EnergyNetController",
        controller_module="energy_net.controllers",
        pricing_policy=PricingPolicy.CONSTANT,  # Use constant pricing policy
        cost_type=CostType.CONSTANT,  # Use constant cost function
        demand_pattern=DemandPattern.SINUSOIDAL  # Use sinusoidal demand pattern
    )
    
    # Reset the environment
    observations, info = env.reset()
    print("Initial observations:", observations)
    
    # Run for 10 steps with random actions
    for step in range(10):
        # Sample random actions for both agents
        actions = {
            "iso": env.action_space["iso"].sample(),
            "pcs": env.action_space["pcs"].sample()
        }
        
        # Take a step in the environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Print step results
        print(f"\nStep {step + 1}:")
        print("Actions:", actions)
        print("Observations:", observations)
        print("Rewards:", rewards)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        
        # Check if episode is done
        if any(terminated.values()) or any(truncated.values()):
            print("Episode finished!")
            break
    
    # Get final metrics
    metrics = env.get_metrics()
    print("\nFinal metrics:", metrics)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main() 