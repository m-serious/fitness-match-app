import numpy as np
import psycopg2
import json
import logging
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity
from embedding import UserProfile, FitnessEmbeddingGenerator, generate_sample_users

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserDatabase:
    """User profile and embedding database manager using PostgreSQL"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string or os.getenv("POSTGRES_URL")
        if not self.connection_string:
            raise ValueError("Need to provide POSTGRES_URL environment variable or connection_string parameter")
        
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fitness_users (
                user_id VARCHAR(255) PRIMARY KEY,
                age INTEGER NOT NULL,
                gender VARCHAR(50) NOT NULL,
                height FLOAT NOT NULL,
                weight FLOAT NOT NULL,
                experience INTEGER NOT NULL,
                body_fat FLOAT,
                frequency INTEGER,
                eat_out_freq VARCHAR(10) NOT NULL,
                cook_freq VARCHAR(10) NOT NULL,
                daily_snacks VARCHAR(10) NOT NULL,
                snack_type VARCHAR(100) NOT NULL,
                fruit_veg_servings VARCHAR(10) NOT NULL,
                beverage_choice VARCHAR(100) NOT NULL,
                diet_preference VARCHAR(50) NOT NULL,
                fitness_goals JSONB NOT NULL,
                struggling_with TEXT DEFAULT '',
                embedding JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("PostgreSQL database initialized successfully")
    
    def add_user(self, profile: UserProfile, embedding: List[float]):
        """
        Add user profile and embedding to database
        
        Args:
            profile: User fitness profile
            embedding: User embedding vector
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert list to JSON for storage
        fitness_goals_json = json.dumps(profile.fitness_goals)
        embedding_json = json.dumps(embedding)
        
        cursor.execute('''
            INSERT INTO fitness_users 
            (user_id, age, gender, height, weight, experience, body_fat, frequency,
             eat_out_freq, cook_freq, daily_snacks, snack_type,
             fruit_veg_servings, beverage_choice, diet_preference,
             fitness_goals, struggling_with, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                age = EXCLUDED.age,
                gender = EXCLUDED.gender,
                height = EXCLUDED.height,
                weight = EXCLUDED.weight,
                experience = EXCLUDED.experience,
                body_fat = EXCLUDED.body_fat,
                frequency = EXCLUDED.frequency,
                eat_out_freq = EXCLUDED.eat_out_freq,
                cook_freq = EXCLUDED.cook_freq,
                daily_snacks = EXCLUDED.daily_snacks,
                snack_type = EXCLUDED.snack_type,
                fruit_veg_servings = EXCLUDED.fruit_veg_servings,
                beverage_choice = EXCLUDED.beverage_choice,
                diet_preference = EXCLUDED.diet_preference,
                fitness_goals = EXCLUDED.fitness_goals,
                struggling_with = EXCLUDED.struggling_with,
                embedding = EXCLUDED.embedding
        ''', (
            profile.user_id, profile.age, profile.gender, profile.height, profile.weight, profile.experience,
            profile.body_fat, profile.frequency, profile.eat_out_freq,
            profile.cook_freq, profile.daily_snacks, profile.snack_type,
            profile.fruit_veg_servings, profile.beverage_choice, profile.diet_preference,
            fitness_goals_json, profile.struggling_with, embedding_json
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"User {profile.user_id} added to database")
    
    def get_all_users_with_embeddings(self) -> List[Tuple[UserProfile, List[float]]]:
        """
        Get all users with their embeddings from database
        
        Returns:
            List of tuples containing (UserProfile, embedding)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM fitness_users ORDER BY created_at')
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        users_with_embeddings = []
        for row in rows:
            try:
                # 数据库字段顺序：
                # 0: user_id, 1: age, 2: gender, 3: height, 4: weight, 5: experience,
                # 6: body_fat, 7: frequency, 8: eat_out_freq, 9: cook_freq, 10: daily_snacks,
                # 11: snack_type, 12: fruit_veg_servings, 13: beverage_choice, 14: diet_preference,
                # 15: fitness_goals, 16: struggling_with, 17: embedding, 18: created_at
                
                # Handle fitness_goals
                fitness_goals = row[15]
                if isinstance(fitness_goals, str):
                    if fitness_goals.strip():
                        fitness_goals = json.loads(fitness_goals)
                    else:
                        fitness_goals = []
                elif isinstance(fitness_goals, list):
                    fitness_goals = fitness_goals
                elif fitness_goals is None:
                    fitness_goals = []
                else:
                    logger.warning(f"Unexpected fitness_goals type for user {row[0]}: {type(fitness_goals)}")
                    fitness_goals = []
                
                # Handle embedding
                embedding = row[17]
                if isinstance(embedding, str):
                    if embedding.strip():
                        embedding = json.loads(embedding)
                    else:
                        logger.error(f"Empty embedding string for user {row[0]}, skipping...")
                        continue
                elif isinstance(embedding, list):
                    embedding = embedding
                elif embedding is None:
                    logger.error(f"NULL embedding for user {row[0]}, skipping...")
                    continue
                else:
                    logger.warning(f"Unexpected embedding type for user {row[0]}: {type(embedding)}")
                    continue
                
                # Validate embedding
                if not embedding or len(embedding) == 0:
                    logger.error(f"Empty embedding vector for user {row[0]}, skipping...")
                    continue
                
                profile = UserProfile(
                    user_id=row[0],
                    age=row[1],
                    gender=row[2],
                    height=row[3],
                    weight=row[4],
                    experience=row[5],
                    body_fat=row[6],
                    frequency=row[7],
                    eat_out_freq=row[8],
                    cook_freq=row[9],
                    daily_snacks=row[10],
                    snack_type=row[11],
                    fruit_veg_servings=row[12],
                    beverage_choice=row[13],
                    diet_preference=row[14],
                    fitness_goals=fitness_goals,
                    struggling_with=row[16] or ""
                )
                users_with_embeddings.append((profile, embedding))
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for user {row[0]}: {e}")
                logger.error(f"Problematic fitness_goals: {repr(row[15])}")
                logger.error(f"Problematic embedding: {repr(row[17])}")
                continue
            except Exception as e:
                logger.error(f"Error processing user {row[0]}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(users_with_embeddings)} users from database")
        return users_with_embeddings
    
    def get_user_count(self) -> int:
        """Get total number of users in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM fitness_users')
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def cleanup_database(self):
        """Clean up and fix data format issues in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        logger.info("Cleaning up database data format...")
        
        # Get all users
        cursor.execute('SELECT user_id, fitness_goals, embedding FROM fitness_users')
        rows = cursor.fetchall()
        
        for user_id, fitness_goals, embedding in rows:
            try:
                # Fix fitness_goals format
                if isinstance(fitness_goals, list):
                    fitness_goals_json = json.dumps(fitness_goals)
                elif isinstance(fitness_goals, str):
                    # Try to parse and re-serialize to ensure valid JSON
                    parsed_goals = json.loads(fitness_goals)
                    fitness_goals_json = json.dumps(parsed_goals)
                else:
                    logger.warning(f"Invalid fitness_goals format for {user_id}, setting to empty list")
                    fitness_goals_json = json.dumps([])
                
                # Fix embedding format
                if isinstance(embedding, list):
                    embedding_json = json.dumps(embedding)
                elif isinstance(embedding, str):
                    # Try to parse and re-serialize to ensure valid JSON
                    parsed_embedding = json.loads(embedding)
                    embedding_json = json.dumps(parsed_embedding)
                else:
                    logger.error(f"Invalid embedding format for {user_id}, skipping...")
                    continue
                
                # Update the record
                cursor.execute('''
                    UPDATE fitness_users 
                    SET fitness_goals = %s, embedding = %s 
                    WHERE user_id = %s
                ''', (fitness_goals_json, embedding_json, user_id))
                
            except Exception as e:
                logger.error(f"Failed to fix data for user {user_id}: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database cleanup completed")
    
    def clear_database(self):
        """Clear all user data from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM fitness_users')
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database cleared")

class FitnessUserMatcher:
    """Fitness user matching system"""
    
    def __init__(self, connection_string: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize user matcher
        
        Args:
            connection_string: PostgreSQL connection string
            api_key: DeepInfra API key for embedding generation
        """
        self.database = UserDatabase(connection_string)
        self.embedding_generator = FitnessEmbeddingGenerator(api_key)
    
    def add_user_to_database(self, profile: UserProfile):
        """
        Add new user to database with generated embedding
        
        Args:
            profile: User fitness profile
        """
        # Generate embedding for the user
        embedding = self.embedding_generator.generate_profile_embedding(profile)
        
        # Add to database
        self.database.add_user(profile, embedding)
        logger.info(f"User {profile.user_id} successfully added to database")
    
    def populate_sample_data(self, force_refresh: bool = False):
        """
        Populate database with sample users if empty
        
        Args:
            force_refresh: If True, clear existing data and repopulate
        """
        if force_refresh:
            logger.info("Force refresh requested - clearing database...")
            self.database.clear_database()
        
        if self.database.get_user_count() == 0:
            logger.info("Database is empty. Generating and adding sample users...")
            sample_users = generate_sample_users()
            
            for user in sample_users:
                try:
                    self.add_user_to_database(user)
                except Exception as e:
                    logger.error(f"Failed to add user {user.user_id}: {str(e)}")
            
            logger.info(f"Added {len(sample_users)} sample users to database")
        else:
            # Try to fix existing data format issues
            try:
                logger.info(f"Database contains {self.database.get_user_count()} users")
                self.database.cleanup_database()
            except Exception as e:
                logger.warning(f"Database cleanup failed: {e}")
                logger.info("Attempting to clear and repopulate database...")
                self.database.clear_database()
                sample_users = generate_sample_users()
                for user in sample_users:
                    try:
                        self.add_user_to_database(user)
                    except Exception as e:
                        logger.error(f"Failed to add user {user.user_id}: {str(e)}")
                logger.info(f"Re-populated database with {len(sample_users)} sample users")
    
    def find_best_matches(self, new_user: UserProfile, top_k: int = 5) -> List[Tuple[UserProfile, float]]:
        """
        Find best matching users for a new user
        
        Args:
            new_user: New user profile to match
            top_k: Number of top matches to return
            
        Returns:
            List of tuples containing (matched_user_profile, similarity_score)
        """
        # Generate embedding for new user
        logger.info(f"Generating embedding for new user {new_user.user_id}")
        new_user_embedding = self.embedding_generator.generate_profile_embedding(new_user)
        
        # Get all existing users with embeddings
        existing_users = self.database.get_all_users_with_embeddings()
        
        if not existing_users:
            logger.warning("No existing users in database for matching")
            return []
        
        logger.info(f"Found {len(existing_users)} existing users for matching")
        
        # Calculate similarities
        similarities = []
        new_user_embedding_array = np.array(new_user_embedding).reshape(1, -1)
        
        for profile, embedding in existing_users:
            # Skip if it's the same user
            if profile.user_id == new_user.user_id:
                continue
                
            existing_embedding_array = np.array(embedding).reshape(1, -1)
            similarity = cosine_similarity(new_user_embedding_array, existing_embedding_array)[0][0]
            similarities.append((profile, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]
        
        logger.info(f"Found {len(top_matches)} matches for user {new_user.user_id}")
        return top_matches
    
    def get_match_details(self, matches: List[Tuple[UserProfile, float]]) -> Dict:
        """
        Get detailed information about matches
        
        Args:
            matches: List of matched users with similarity scores
            
        Returns:
            Dictionary containing match details
        """
        match_details = {
            "total_matches": len(matches),
            "matches": []
        }
        
        for i, (profile, similarity) in enumerate(matches):
            match_info = {
                "rank": i + 1,
                "user_id": profile.user_id,
                "similarity_score": round(similarity, 4),
                "age": profile.age,
                "gender": profile.gender,
                "height": profile.height,
                "weight": profile.weight,
                "experience": profile.experience,
                "fitness_goals": profile.fitness_goals,
                "diet_preference": profile.diet_preference,
                "struggling_with": profile.struggling_with
            }
            match_details["matches"].append(match_info)
        
        return match_details

def main():
    """Usage example and testing"""
    try:
        # Initialize matcher
        matcher = FitnessUserMatcher()
        
        # Populate with sample data if database is empty
        matcher.populate_sample_data()
        
        # Create new user for matching
        new_user = UserProfile(
            user_id="new_user_001",
            age=25,
            gender="Female",
            height=170.0,
            weight=70.0,
            experience=2,
            eat_out_freq="2–3",
            cook_freq="4–5",
            daily_snacks="1",
            snack_type="Healthy (fruit/nuts)",
            fruit_veg_servings="4–5",
            beverage_choice="Mostly water",
            diet_preference="Omnivore",
            fitness_goals=["Weight Loss", "Cardio Fitness"],
            body_fat=18.5,
            frequency=4,
            struggling_with="Finding time for consistent workouts"
        )
        
        # Find best matches
        logger.info("Finding best matches for new user...")
        matches = matcher.find_best_matches(new_user, top_k=3)
        
        # Display results
        match_details = matcher.get_match_details(matches)
        
        print("\n" + "="*80)
        print("FITNESS MATCHING RESULTS")
        print("="*80)
        print(f"New User: {new_user.user_id}")
        print(f"Profile: {new_user.height}cm, {new_user.weight}kg, {new_user.experience}yr exp")
        print(f"Goals: {', '.join(new_user.fitness_goals)}")
        print(f"Diet: {new_user.diet_preference}")
        print(f"Total matches found: {match_details['total_matches']}")
        print("\nTop Matches:")
        print("-"*80)
        
        for match in match_details['matches']:
            print(f"Rank #{match['rank']}")
            print(f"User ID: {match['user_id']}")
            print(f"Similarity Score: {match['similarity_score']:.4f}")
            print(f"Profile: {match['height']}cm, {match['weight']}kg, {match['experience']}yr exp")
            print(f"Goals: {', '.join(match['fitness_goals'])}")
            print(f"Diet: {match['diet_preference']}")
            if match['struggling_with']:
                print(f"Struggling: {match['struggling_with']}")
            print("-"*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()