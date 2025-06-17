import os
import logging
import psycopg2
import json
from typing import List, Union, Optional, Dict
from openai import OpenAI
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User fitness profile data structure"""
    username: str
    
    # Part 1: Basic Information (required fields first)
    age: int  # in years - required
    gender: str  # "Male", "Female", "Non-binary", "Prefer not to say" - required
    location: str  # User's location/city - required
    height: float  # in cm - required
    weight: float  # in kg - required
    experience: int  # years of fitness experience - required
    
    # Part 2: Diet Habits (required single choice questions)
    eat_out_freq: str  # "0–1", "2–3", "4–5", "6+"
    cook_freq: str  # "0–1", "2–3", "4–5", "6+"
    daily_snacks: str  # "0", "1", "2", "3+"
    snack_type: str  # "Sweet (candy/dessert)", "Salty (chips/crackers)", "Healthy (fruit/nuts)", "Mixed / varies"
    fruit_veg_servings: str  # "0–1", "2–3", "4–5", "6+"
    beverage_choice: str  # "Mostly water", "Water & diet drinks", "Sugary drinks daily", "Mostly coffee/tea"
    diet_preference: str  # "Omnivore", "Vegetarian", "Vegan", "Keto / low-carb"
    
    # Part 3: Fitness Goals (required)
    fitness_goals: List[str]  # At least one from predefined list - required
    
    # Optional fields (must come after all required fields)
    body_fat: Optional[float] = None  # percentage - optional
    frequency: Optional[int] = None  # times per week - optional
    struggling_with: str = ""  # Optional string field

class FitnessEmbeddingGenerator:
    """Fitness user profile embedding generator"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            api_key: DeepInfra API key, if None will get from environment variable
        """
        self.api_key = api_key or os.getenv("DEEPINFRA_TOKEN")
        if not self.api_key:
            raise ValueError("Need to provide DEEPINFRA_TOKEN environment variable or api_key parameter")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = "Qwen/Qwen3-Embedding-4B"
    
    def profile_to_text(self, profile: UserProfile) -> str:
        """
        Convert user profile to text description
        
        Args:
            profile: User fitness profile
            
        Returns:
            Formatted text description
        """
        # Basic Information
        basic_info = f"Age: {profile.age} years, Gender: {profile.gender}, Location: {profile.location}, Height: {profile.height}cm, Weight: {profile.weight}kg, Experience: {profile.experience} years"
        if profile.body_fat:
            basic_info += f", Body Fat: {profile.body_fat}%"
        if profile.frequency:
            basic_info += f", Workout Frequency: {profile.frequency} times per week"
        
        # Diet Habits
        diet_info = f"""
        Diet Habits:
        - Eats out: {profile.eat_out_freq} times per week
        - Cooks: {profile.cook_freq} times per week
        - Daily snacks: {profile.daily_snacks}
        - Snack type: {profile.snack_type}
        - Fruit & vegetable servings: {profile.fruit_veg_servings} daily
        - Beverage preference: {profile.beverage_choice}
        - Diet preference: {profile.diet_preference}
        """
        
        # Fitness Goals
        goals_info = f"Fitness Goals: {', '.join(profile.fitness_goals)}"
        
        text = f"""
        Basic Information: {basic_info}
        {diet_info.strip()}
        {goals_info}
        """
        
        if profile.struggling_with:
            text += f"\nStruggling with: {profile.struggling_with}"
        
        return text.strip()
    
    def generate_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding vectors for text
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Embedding vector or list of vectors
        """
        try:
            logger.info(f"Generating embedding, input type: {'single text' if isinstance(text, str) else f'text list ({len(text)} items)'}")
            
            embeddings = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            logger.info(f"Successfully generated embedding, tokens used: {embeddings.usage.prompt_tokens}")
            
            if isinstance(text, str):
                return embeddings.data[0].embedding
            else:
                return [embeddings.data[i].embedding for i in range(len(text))]
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def generate_profile_embedding(self, profile: UserProfile) -> List[float]:
        """
        Generate embedding for user fitness profile
        
        Args:
            profile: User fitness profile
            
        Returns:
            Profile embedding vector
        """
        profile_text = self.profile_to_text(profile)
        logger.info(f"Generating profile embedding for user {profile.username}")
        return self.generate_embedding(profile_text)
    
    def generate_batch_embeddings(self, profiles: List[UserProfile]) -> List[List[float]]:
        """
        Generate embeddings for multiple user profiles in batch
        
        Args:
            profiles: List of user profiles
            
        Returns:
            List of embedding vectors
        """
        profile_texts = [self.profile_to_text(profile) for profile in profiles]
        logger.info(f"Batch generating embeddings for {len(profiles)} user profiles")
        return self.generate_embedding(profile_texts)

def generate_sample_users() -> List[UserProfile]:
    """Generate 10 sample users with random but realistic data"""
    import random
    
    # Predefined options for survey questions
    freq_options = ["0–1", "2–3", "4–5", "6+"]
    snacks_options = ["0", "1", "2", "3+"]
    snack_types = ["Sweet (candy/dessert)", "Salty (chips/crackers)", "Healthy (fruit/nuts)", "Mixed / varies"]
    beverages = ["Mostly water", "Water & diet drinks", "Sugary drinks daily", "Mostly coffee/tea"]
    diet_prefs = ["Omnivore", "Vegetarian", "Vegan", "Keto / low-carb"]
    genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
    
    # Sample locations
    locations = [
        "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
        "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
        "Austin, TX", "Jacksonville, FL", "San Francisco, CA", "Columbus, OH", "Charlotte, NC",
        "Fort Worth, TX", "Indianapolis, IN", "Seattle, WA", "Denver, CO", "Boston, MA",
        "Miami, FL", "Atlanta, GA", "Portland, OR", "Las Vegas, NV", "Detroit, MI"
    ]

    fitness_goals_options = [
        "Weight Loss", "Muscle Gain", "Strength Training", "Cardio Fitness",
        "Flexibility", "Endurance", "General Health", "Sports Performance"
    ]
    
    struggling_examples = [
        "Finding time for consistent workouts",
        "Staying motivated during winter months",
        "Controlling portion sizes",
        "Building upper body strength",
        "Maintaining workout routine while traveling",
        "Balancing cardio and strength training",
        "Eating healthy on a budget",
        "Recovering from previous injury",
        "Setting realistic fitness goals",
        ""  # Some users might not fill this
    ]
    
    sample_users = []
    
    for i in range(1, 11):
        # Generate realistic basic info
        age = random.randint(15, 65)  # 15-65 years old
        gender = random.choice(genders)
        location = random.choice(locations)
        
        # Height varies by gender for realism
        if gender == "Male":
            height = round(random.uniform(155, 195), 1)  # 165-195cm for males
        elif gender == "Female":
            height = round(random.uniform(145, 185), 1)  # 155-185cm for females
        else:
            height = round(random.uniform(145, 195), 1)  # Full range for others

        weight = round(random.uniform(40, 120), 1)  # 50-120kg
        experience = random.randint(0, 10)  # 0-10 years
        
        # Required diet survey answers
        eat_out = random.choice(freq_options)
        cook_freq = random.choice(freq_options)
        daily_snacks = random.choice(snacks_options)
        snack_type = random.choice(snack_types)
        fruit_veg = random.choice(freq_options)
        beverage = random.choice(beverages)
        diet_pref = random.choice(diet_prefs)
        
        # Fitness goals (1-3 goals per user)
        num_goals = random.randint(1, 3)
        goals = random.sample(fitness_goals_options, num_goals)
        
        # Optional fields (70% chance to fill)
        body_fat = round(random.uniform(8, 35), 1) if random.random() < 0.7 else None
        frequency = random.randint(1, 7) if random.random() < 0.7 else None
        struggling = random.choice(struggling_examples)
        
        user = UserProfile(
            username=f"user_{i:04d}",
            age=age,
            gender=gender,
            location=location,
            height=height,
            weight=weight,
            experience=experience,
            eat_out_freq=eat_out,
            cook_freq=cook_freq,
            daily_snacks=daily_snacks,
            snack_type=snack_type,
            fruit_veg_servings=fruit_veg,
            beverage_choice=beverage,
            diet_preference=diet_pref,
            fitness_goals=goals,
            body_fat=body_fat,
            frequency=frequency,
            struggling_with=struggling
        )
        
        sample_users.append(user)
    
    return sample_users

# Usage example
def main():
    """Usage example"""
    try:
        # Initialize embedding generator
        generator = FitnessEmbeddingGenerator()
        
        # Generate sample users
        sample_users = generate_sample_users()
        
        print("Generated Sample Users:")
        print("=" * 50)
        
        for user in sample_users:
            print(f"\nUser ID: {user.username}")
            print(f"Basic: {user.height}cm, {user.weight}kg, {user.experience}yr exp")
            if user.body_fat:
                print(f"Body Fat: {user.body_fat}%")
            if user.frequency:
                print(f"Frequency: {user.frequency}x/week")
            print(f"Goals: {', '.join(user.fitness_goals)}")
            print(f"Diet: {user.diet_preference}, Eats out: {user.eat_out_freq}")
            if user.struggling_with:
                print(f"Struggling: {user.struggling_with}")
        
        # Generate embedding for first user as example
        print(f"\nGenerating embedding for {sample_users[0].username}...")
        embedding = generator.generate_profile_embedding(sample_users[0])
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")

if __name__ == "__main__":
    main()