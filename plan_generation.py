import os
import logging
import json
from typing import List, Dict, Optional
from openai import OpenAI
from dataclasses import dataclass
from embedding import UserProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitnessPlanGenerator:
    """Generate personalized fitness plans using OpenAI GPT-4o-mini"""
    
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
    
    def create_user_summary(self, primary_user: UserProfile, matched_user: UserProfile) -> str:
        """
        Create a detailed summary of both user profiles
        
        Args:
            primary_user: Primary user profile
            matched_user: Matched partner profile
            
        Returns:
            Formatted users summary
        """
        primary_info = f"""
Primary User: {primary_user.username}
- Age: {primary_user.age}, Gender: {primary_user.gender}, Location: {primary_user.location}
- Physical: {primary_user.height}cm, {primary_user.weight}kg, Experience: {primary_user.experience} years
- Goals: {', '.join(primary_user.fitness_goals)}
- Diet: {primary_user.diet_preference}
- Frequency: {primary_user.frequency}x/week
- Struggles: {primary_user.struggling_with}
        """
        
        matched_info = f"""
Matched User: {matched_user.username}
- Age: {matched_user.age}, Gender: {matched_user.gender}, Location: {matched_user.location}
- Physical: {matched_user.height}cm, {matched_user.weight}kg, Experience: {matched_user.experience} years
- Goals: {', '.join(matched_user.fitness_goals)}
- Diet: {matched_user.diet_preference}
- Frequency: {matched_user.frequency}x/week
- Struggles: {matched_user.struggling_with}
        """
        
        return f"{primary_info}\n{matched_info}"
    
    def generate_fitness_group_prompt(self, primary_user: UserProfile, matched_user: UserProfile) -> str:
        """
        Generate comprehensive prompt for fitness group creation
        
        Args:
            primary_user: Primary user seeking fitness plan
            matched_user: Matched partner user
            
        Returns:
            Detailed prompt for AI model to generate JSON fitness group plan
        """
        users_summary = self.create_user_summary(primary_user, matched_user)
        
        # Determine shared location for group name
        shared_location = primary_user.location.split(',')[0] if primary_user.location == matched_user.location else "Mixed Location"
        
        # Find common goals
        common_goals = list(set(primary_user.fitness_goals) & set(matched_user.fitness_goals))
        primary_goal = common_goals[0] if common_goals else primary_user.fitness_goals[0]
        
        prompt = f"""
You are a professional fitness coach creating a structured fitness group for two matched users. Based on their profiles, create a comprehensive fitness group plan.

USER PROFILES:
{users_summary}

REQUIREMENTS:
You must return a valid JSON object with the following exact structure. Each field must be filled with specific, actionable content:

{{
    "groupId": "group_[generate_unique_id]",
    "groupName": "[Create catchy name based on their location, time preference, or main activity]",
    "description": "[2-3 sentences describing the group's focus and what makes it special]",
    "goal": "[Primary shared fitness goal from their profiles]",
    "weeklyPlan": {{
        "howManyWeeks": "[Number of weeks for the plan, typically 4-8]",
        "oddDayWorkoutPlan": {{
            "title": "[Name for odd day workouts (Mon/Wed/Fri)]",
            "duration": "[Total workout time]",
            "exercises": ["[Exercise 1]", "[Exercise 2]", "[Exercise 3]", "[Exercise 4]", "[Exercise 5]"],
            "diet": "[Specific dietary recommendations for odd day workouts]"
        }},
        "evenDayWorkoutPlan": {{
            "title": "[Name for even day workouts (Tue/Thu/Sat)]",
            "duration": "[Total workout time]",
            "exercises": ["[Exercise 1]", "[Exercise 2]", "[Exercise 3]", "[Exercise 4]", "[Exercise 5]"],
            "diet": "[Specific dietary recommendations for even day workouts]"
        }}
    }},
    "memberFullNames": ["{primary_user.username}", "{matched_user.username}"],
    "memberUsernames": ["{primary_user.username}", "{matched_user.username}"]
}}

IMPORTANT GUIDELINES:
1. Consider both users' experience levels ({primary_user.experience} vs {matched_user.experience} years)
2. Account for their shared location: {shared_location}
3. Focus on their common goal: {primary_goal}
4. Make exercises appropriate for both fitness levels
5. Consider their diet preferences: {primary_user.diet_preference} and {matched_user.diet_preference}
6. Address their struggles: "{primary_user.struggling_with}" and "{matched_user.struggling_with}"
7. Odd days should focus on one aspect (e.g., strength, cardio)
8. Even days should complement odd days (e.g., if odd=strength, even=cardio)
9. Return ONLY the JSON object, no additional text or formatting
10. Ensure all JSON syntax is correct with proper quotes and commas

Generate a unique group ID, creative group name, and comprehensive workout plans that both users can follow together effectively.
        """
        return prompt
    
    def generate_fitness_group_plan(self, primary_user: UserProfile, matched_user: UserProfile) -> Dict:
        """
        Generate a comprehensive fitness group plan in JSON format
        
        Args:
            primary_user: Primary user profile
            matched_user: Matched partner profile
            
        Returns:
            Dictionary containing the generated fitness group plan
        """
        try:
            logger.info(f"Generating fitness group plan for {primary_user.username} and {matched_user.username}")
            
            # Create the prompt
            prompt = self.generate_fitness_group_prompt(primary_user, matched_user)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional fitness coach. Return only valid JSON objects for fitness group plans. Do not include any markdown formatting or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Get the response content
            response_content = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            response_content = response_content.strip()
            
            # Parse JSON response
            try:
                fitness_group_plan = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {response_content}")
                raise ValueError(f"Invalid JSON response from GPT: {e}")
            
            # Add metadata
            fitness_group_plan["metadata"] = {
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "N/A",
                "primary_user": primary_user.username,
                "matched_user": matched_user.username
            }
            
            logger.info("Successfully generated fitness group plan")
            return fitness_group_plan
            
        except Exception as e:
            logger.error(f"Failed to generate fitness group plan: {str(e)}")
            raise
    
    def save_fitness_plan(self, plan_data: Dict, filename: Optional[str] = None) -> str:
        """
        Save the generated fitness plan to a JSON file
        
        Args:
            plan_data: Dictionary containing plan data
            filename: Optional filename, if None will auto-generate
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            group_id = plan_data.get('groupId', 'unknown_group')
            filename = f"fitness_group_plan_{group_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Fitness group plan saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save fitness group plan: {str(e)}")
            raise

def main():
    """Usage example and testing"""
    try:
        # Initialize plan generator
        generator = FitnessPlanGenerator()
        
        # Create sample user profiles
        primary_user = UserProfile(
            username="alice_runner",
            age=28,
            gender="Female",
            location="San Francisco, CA",
            height=170.0,
            weight=65.0,
            experience=2,
            eat_out_freq="2–3",
            cook_freq="4–5",
            daily_snacks="1",
            snack_type="Healthy (fruit/nuts)",
            fruit_veg_servings="4–5",
            beverage_choice="Mostly water",
            diet_preference="Omnivore",
            fitness_goals=["Weight Loss", "Cardio Fitness"],
            body_fat=20.0,
            frequency=4,
            struggling_with="Finding time for consistent workouts"
        )
        
        matched_user = UserProfile(
            username="bob_lifter",
            age=32,
            gender="Male",
            location="San Francisco, CA",
            height=180.0,
            weight=75.0,
            experience=3,
            eat_out_freq="1–2",
            cook_freq="5–6",
            daily_snacks="0",
            snack_type="Healthy (fruit/nuts)",
            fruit_veg_servings="4–5",
            beverage_choice="Mostly water",
            diet_preference="Omnivore",
            fitness_goals=["Cardio Fitness", "Strength Training"],
            body_fat=15.0,
            frequency=5,
            struggling_with="Balancing cardio and strength training"
        )
        
        # Generate fitness group plan
        logger.info("Generating fitness group plan...")
        group_plan = generator.generate_fitness_group_plan(primary_user, matched_user)
        
        # Display plan summary
        print("\n" + "="*80)
        print("FITNESS GROUP PLAN GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Group ID: {group_plan['groupId']}")
        print(f"Group Name: {group_plan['groupName']}")
        print(f"Description: {group_plan['description']}")
        print(f"Goal: {group_plan['goal']}")
        print(f"Duration: {group_plan['weeklyPlan']['howManyWeeks']} weeks")
        print(f"Members: {', '.join(group_plan['memberFullNames'])}")
        print(f"Model Used: {group_plan['metadata']['model_used']}")
        print(f"Tokens Used: {group_plan['metadata']['tokens_used']}")
        
        # Display workout plans
        print("\n" + "="*80)
        print("WEEKLY WORKOUT PLAN")
        print("="*80)
        
        odd_plan = group_plan['weeklyPlan']['oddDayWorkoutPlan']
        print(f"\nODD DAYS (Mon/Wed/Fri): {odd_plan['title']}")
        print(f"Duration: {odd_plan['duration']}")
        print("Exercises:")
        for i, exercise in enumerate(odd_plan['exercises'], 1):
            print(f"  {i}. {exercise}")
        print(f"Diet: {odd_plan['diet']}")
        
        even_plan = group_plan['weeklyPlan']['evenDayWorkoutPlan']
        print(f"\nEVEN DAYS (Tue/Thu/Sat): {even_plan['title']}")
        print(f"Duration: {even_plan['duration']}")
        print("Exercises:")
        for i, exercise in enumerate(even_plan['exercises'], 1):
            print(f"  {i}. {exercise}")
        print(f"Diet: {even_plan['diet']}")
        
        # Save to file
        saved_file = generator.save_fitness_plan(group_plan)
        print(f"\nGroup plan saved to: {saved_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()