import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dataclasses import dataclass
from embedding import UserProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitnessPlanGenerator:
    """Generate personalized fitness plans using OpenAI o4-mini"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize fitness plan generator
        
        Args:
            api_key: OpenAI API key, if None will get from environment variable
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Need to provide OPENAI_API_KEY environment variable or api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini-2024-07-18"
    
    def create_user_summary(self, profile: UserProfile, is_primary: bool = True) -> str:
        """
        Create a detailed summary of user profile
        
        Args:
            profile: User fitness profile
            is_primary: Whether this is the primary user or matched partner
            
        Returns:
            Formatted user summary
        """
        role = "Primary User" if is_primary else "Matched Partner"
        
        # Basic Information
        basic_info = f"Age: {profile.age} years, Gender: {profile.gender}, Height: {profile.height}cm, Weight: {profile.weight}kg, Experience: {profile.experience} years"
        if profile.body_fat:
            basic_info += f", Body Fat: {profile.body_fat}%"
        if profile.frequency:
            basic_info += f", Workout Frequency: {profile.frequency} times per week"
        
        # Diet Summary
        diet_summary = f"""
        Diet Habits:
        - Eating out: {profile.eat_out_freq} times/week
        - Cooking: {profile.cook_freq} times/week
        - Daily snacks: {profile.daily_snacks}
        - Snack preference: {profile.snack_type}
        - Fruit & vegetable intake: {profile.fruit_veg_servings} servings/day
        - Beverage preference: {profile.beverage_choice}
        - Diet type: {profile.diet_preference}
        """
        
        summary = f"""
{role}: {profile.user_id}
Basic Information: {basic_info}
{diet_summary.strip()}
Fitness Goals: {', '.join(profile.fitness_goals)}
        """
        
        if profile.struggling_with:
            summary += f"\nCurrent Struggles: {profile.struggling_with}"
        
        return summary.strip()
    
    def generate_fitness_plan_prompt(self, primary_user: UserProfile, matched_user: UserProfile) -> str:
        """
        Generate comprehensive prompt for fitness plan creation
        
        Args:
            primary_user: Primary user seeking fitness plan
            matched_user: Matched partner user
            
        Returns:
            Detailed prompt for AI model
        """
        primary_summary = self.create_user_summary(primary_user, is_primary=True)
        matched_summary = self.create_user_summary(matched_user, is_primary=False)
        
        prompt = f"""
You are a professional fitness coach and nutritionist. Create a comprehensive, personalized fitness and health plan for two matched users who want to work out together.

USER PROFILES:
{primary_summary}

{matched_summary}

REQUIREMENTS:
Please create a detailed fitness plan that includes:

1. COMPATIBILITY ANALYSIS
   - Analyze how these two users complement each other based on their experience levels, goals, and preferences
   - Identify shared fitness goals and compatible workout styles
   - Note any differences in experience/fitness level that need accommodation
   - Consider their diet preferences and how they can support each other

2. WEEKLY WORKOUT SCHEDULE
   - Design a 4-6 week progressive plan considering both users' experience levels
   - Include specific exercises for each day that work for both fitness levels
   - Account for their stated workout frequency preferences
   - Suggest modifications for different experience levels
   - Include proper rest days and recovery periods

3. PARTNER WORKOUT ROUTINES
   - Specific exercises they can do together effectively
   - How to motivate and support each other during workouts
   - Safety considerations when working out as a team with different experience levels
   - Partner-based exercises that can accommodate different fitness levels

4. INDIVIDUAL FOCUS AREAS
   - Address each user's specific fitness goals separately
   - Exercises tailored to their individual experience levels and goals
   - Consider what each user is currently struggling with

5. NUTRITION GUIDELINES
   - Consider their different diet preferences and eating habits
   - Meal timing recommendations around their workout schedule
   - Hydration recommendations for their activity level
   - How they can support each other's dietary goals despite different preferences
   - Address their current eating patterns (cooking frequency, eating out, snacking habits)

6. PROGRESS TRACKING
   - Key metrics appropriate for their experience levels and goals
   - How to track progress together while respecting individual differences
   - When and how to adjust the plan as they progress
   - Benchmarks appropriate for their current fitness levels

7. MOTIVATION AND ACCOUNTABILITY
   - Strategies for staying motivated as partners with different experience levels
   - How to handle different progress rates between beginner and more experienced partners
   - Communication tips for fitness partnerships
   - Address their specific struggles mentioned in their profiles

8. SAFETY AND INJURY PREVENTION
   - Warm-up and cool-down routines appropriate for their experience levels
   - Common injury prevention strategies for their chosen activities
   - When to rest or modify exercises, especially important for different experience levels
   - How the more experienced partner can help ensure safety

Please make the plan practical and achievable, considering their experience levels, specific goals, dietary preferences, and current challenges. Provide specific, actionable advice that acknowledges their different starting points while creating opportunities for them to work together effectively.

Format the response in clear sections with specific, actionable advice.
        """
        return prompt
    
    def generate_fitness_plan(self, primary_user: UserProfile, matched_user: UserProfile) -> Dict[str, str]:
        """
        Generate a comprehensive fitness plan for matched users
        
        Args:
            primary_user: Primary user profile
            matched_user: Matched partner profile
            
        Returns:
            Dictionary containing the generated fitness plan and metadata
        """
        try:
            logger.info(f"Generating fitness plan for {primary_user.user_id} and {matched_user.user_id}")
            
            # Create the prompt
            prompt = self.generate_fitness_plan_prompt(primary_user, matched_user)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                # temperature=0.7,
                # max_tokens=4000
            )
            
            fitness_plan = response.choices[0].message.content
            
            logger.info("Successfully generated fitness plan")
            
            # Return structured response
            return {
                "primary_user_id": primary_user.user_id,
                "matched_user_id": matched_user.user_id,
                "fitness_plan": fitness_plan,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "N/A"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate fitness plan: {str(e)}")
            raise
    
    def save_fitness_plan(self, plan_data: Dict[str, str], filename: Optional[str] = None) -> str:
        """
        Save the generated fitness plan to a file
        
        Args:
            plan_data: Dictionary containing plan data
            filename: Optional filename, if None will auto-generate
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"fitness_plan_{plan_data['primary_user_id']}_{plan_data['matched_user_id']}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("PERSONALIZED FITNESS PLAN\n")
                f.write("="*80 + "\n\n")
                f.write(f"Primary User: {plan_data['primary_user_id']}\n")
                f.write(f"Matched Partner: {plan_data['matched_user_id']}\n")
                f.write(f"Generated by: {plan_data['model_used']}\n")
                f.write(f"Tokens Used: {plan_data['tokens_used']}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(plan_data['fitness_plan'])
            
            logger.info(f"Fitness plan saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save fitness plan: {str(e)}")
            raise

def main():
    """Usage example and testing"""
    try:
        # Initialize plan generator
        generator = FitnessPlanGenerator()
        
        # Create sample user profiles (simulate matched users with new structure)
        primary_user = UserProfile(
            user_id="user_001",
            age=28,
            gender="Female",
            height=170.0,
            weight=72.0,
            experience=1,  # 1 year experience (beginner)
            eat_out_freq="2–3",
            cook_freq="4–5",
            daily_snacks="1",
            snack_type="Healthy (fruit/nuts)",
            fruit_veg_servings="4–5",
            beverage_choice="Mostly water",
            diet_preference="Omnivore",
            fitness_goals=["Weight Loss", "Cardio Fitness"],
            body_fat=22.0,
            frequency=3,
            struggling_with="Finding time for consistent workouts and controlling portion sizes"
        )
        
        matched_user = UserProfile(
            user_id="user_002",
            age=32,
            gender="Male",
            height=165.0,
            weight=65.0,
            experience=3,  # 3 years experience (intermediate)
            eat_out_freq="0–1",
            cook_freq="6+",
            daily_snacks="0",
            snack_type="Healthy (fruit/nuts)",
            fruit_veg_servings="6+",
            beverage_choice="Mostly water",
            diet_preference="Vegetarian",
            fitness_goals=["Weight Loss", "Strength Training", "General Health"],
            body_fat=18.5,
            frequency=4,
            struggling_with="Balancing cardio and strength training effectively"
        )
        
        # Generate fitness plan
        logger.info("Generating personalized fitness plan...")
        plan_data = generator.generate_fitness_plan(primary_user, matched_user)
        
        # Display plan summary
        print("\n" + "="*80)
        print("FITNESS PLAN GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Primary User: {plan_data['primary_user_id']}")
        print(f"  - Experience: {primary_user.experience} years")
        print(f"  - Goals: {', '.join(primary_user.fitness_goals)}")
        print(f"  - Diet: {primary_user.diet_preference}")
        print(f"Matched Partner: {plan_data['matched_user_id']}")
        print(f"  - Experience: {matched_user.experience} years")
        print(f"  - Goals: {', '.join(matched_user.fitness_goals)}")
        print(f"  - Diet: {matched_user.diet_preference}")
        print(f"Model Used: {plan_data['model_used']}")
        print(f"Tokens Used: {plan_data['tokens_used']}")
        print("\n" + "="*80)
        print("GENERATED FITNESS PLAN:")
        print("="*80)
        print(plan_data['fitness_plan'])
        
        # Save to file
        saved_file = generator.save_fitness_plan(plan_data)
        print(f"\nPlan saved to: {saved_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()