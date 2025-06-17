import os
import logging
from typing import List, Union, Optional
from openai import OpenAI
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User fitness profile data structure"""
    user_id: str
    goals: str  # Fitness goals
    weight: float
    height: float
    age: int
    fitness_level: str  # Beginner/Intermediate/Advanced
    preferred_activities: List[str]  # Preferred exercise types
    schedule: str  # Training schedule
    location: str  # Geographic location
    additional_info: str = ""  # Additional information

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
        text = f"""
        Fitness Goals: {profile.goals}
        Body Data: Weight {profile.weight}kg, Height {profile.height}cm, Age {profile.age} years
        Fitness Level: {profile.fitness_level}
        Preferred Activities: {', '.join(profile.preferred_activities)}
        Training Schedule: {profile.schedule}
        Location: {profile.location}
        """
        
        if profile.additional_info:
            text += f"Additional Info: {profile.additional_info}"
        
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
        logger.info(f"Generating profile embedding for user {profile.user_id}")
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

# Usage example
def main():
    """Usage example"""
    try:
        # Initialize embedding generator
        generator = FitnessEmbeddingGenerator()
        
        # Create sample user profile
        sample_profile = UserProfile(
            user_id="user_001",
            goals="Fat loss and muscle gain, improve cardiovascular fitness",
            weight=70.5,
            height=175.0,
            age=28,
            fitness_level="Intermediate",
            preferred_activities=["Running", "Weight lifting", "Swimming"],
            schedule="3-4 times per week, evening training",
            location="Beijing Chaoyang District",
            additional_info="Hope to find training partner to stay motivated"
        )
        
        # Generate embedding for single profile
        embedding = generator.generate_profile_embedding(sample_profile)
        print(f"User profile embedding dimension: {len(embedding)}")
        print(f"First 5 embedding values: {embedding[:5]}")
        
        # Batch generation example (if you have multiple users)
        # batch_embeddings = generator.generate_batch_embeddings([sample_profile])
        # print(f"Batch generation completed, total {len(batch_embeddings)} embeddings")
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")

if __name__ == "__main__":
    main()