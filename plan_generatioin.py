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
    """Generate personalized fitness plans using OpenAI o1-mini"""
    
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
        self.model = "o4-mini"
    
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
        
        summary = f"""
{role}: {profile.user_id}
- Goals: {profile.goals}
- Physical Stats: {profile.weight}kg, {profile.height}cm, {profile.age} years old
- Current Fitness Level: {profile.fitness_level}
- Preferred Activities: {', '.join(profile.preferred_activities)}
- Available Schedule: {profile.schedule}
- Location: {profile.location}
- Additional Notes: {profile.additional_info}
        """
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
   - Analyze how these two users complement each other
   - Identify shared goals and activities
   - Note any differences that need accommodation

2. WEEKLY WORKOUT SCHEDULE
   - Design a 4-6 week progressive plan
   - Include specific exercises for each day
   - Account for both users' fitness levels and preferences
   - Suggest modifications for different fitness levels
   - Include rest days and recovery

3. PARTNER WORKOUT ROUTINES
   - Specific exercises they can do together
   - How to motivate and support each other
   - Safety considerations when working out as a team

4. INDIVIDUAL FOCUS AREAS
   - Specific recommendations for each user's unique goals
   - Exercises tailored to their individual needs

5. NUTRITION GUIDELINES
   - General healthy eating principles
   - Meal timing around workouts
   - Hydration recommendations
   - Any specific dietary considerations based on their goals

6. PROGRESS TRACKING
   - Key metrics to monitor
   - How to track progress together
   - When and how to adjust the plan

7. MOTIVATION AND ACCOUNTABILITY
   - Strategies for staying motivated as partners
   - How to handle different progress rates
   - Communication tips for fitness partnerships

8. SAFETY AND INJURY PREVENTION
   - Warm-up and cool-down routines
   - Common injury prevention strategies
   - When to rest or modify exercises

Please make the plan practical, achievable, and tailored to their specific circumstances including location, schedule, and equipment availability.

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
                ],
                temperature=0.7,
                max_tokens=4000
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
        
        # Create sample user profiles (simulate matched users)
        primary_user = UserProfile(
            user_id="user_001",
            goals="Weight loss and improve cardiovascular health",
            weight=72.0,
            height=168.0,
            age=26,
            fitness_level="Beginner",
            preferred_activities=["Running", "Swimming", "Yoga"],
            schedule="Evening workouts, 3-4 times per week",
            location="New York City",
            additional_info="Want to lose 10kg and find workout buddy for motivation"
        )
        
        matched_user = UserProfile(
            user_id="user_002",
            goals="Weight loss and muscle toning",
            weight=65.0,
            height=165.0,
            age=28,
            fitness_level="Beginner",
            preferred_activities=["Yoga", "Running", "Pilates"],
            schedule="Weekend workouts and some weekday evenings",
            location="New York City",
            additional_info="New to fitness, looking for supportive workout partner"
        )
        
        # Generate fitness plan
        logger.info("Generating personalized fitness plan...")
        plan_data = generator.generate_fitness_plan(primary_user, matched_user)
        
        # Display plan summary
        print("\n" + "="*80)
        print("FITNESS PLAN GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Primary User: {plan_data['primary_user_id']}")
        print(f"Matched Partner: {plan_data['matched_user_id']}")
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