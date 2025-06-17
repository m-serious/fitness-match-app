import numpy as np
import sqlite3
import json
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity
from match import UserProfile, FitnessEmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserDatabase:
    """User profile and embedding database manager"""
    
    def __init__(self, db_path: str = "fitness_users.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                goals TEXT,
                weight REAL,
                height REAL,
                age INTEGER,
                fitness_level TEXT,
                preferred_activities TEXT,
                schedule TEXT,
                location TEXT,
                additional_info TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def add_user(self, profile: UserProfile, embedding: List[float]):
        """
        Add user profile and embedding to database
        
        Args:
            profile: User fitness profile
            embedding: User embedding vector
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert list to JSON string for storage
        preferred_activities_json = json.dumps(profile.preferred_activities)
        embedding_json = json.dumps(embedding)
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, goals, weight, height, age, fitness_level, 
             preferred_activities, schedule, location, additional_info, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id, profile.goals, profile.weight, profile.height,
            profile.age, profile.fitness_level, preferred_activities_json,
            profile.schedule, profile.location, profile.additional_info,
            embedding_json
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"User {profile.user_id} added to database")
    
    def get_all_users_with_embeddings(self) -> List[Tuple[UserProfile, List[float]]]:
        """
        Get all users with their embeddings from database
        
        Returns:
            List of tuples containing (UserProfile, embedding)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users')
        rows = cursor.fetchall()
        conn.close()
        
        users_with_embeddings = []
        for row in rows:
            profile = UserProfile(
                user_id=row[0],
                goals=row[1],
                weight=row[2],
                height=row[3],
                age=row[4],
                fitness_level=row[5],
                preferred_activities=json.loads(row[6]),
                schedule=row[7],
                location=row[8],
                additional_info=row[9]
            )
            embedding = json.loads(row[10])
            users_with_embeddings.append((profile, embedding))
        
        return users_with_embeddings
    
    def get_user_count(self) -> int:
        """Get total number of users in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        count = cursor.fetchone()[0]
        conn.close()
        return count

class FitnessUserMatcher:
    """Fitness user matching system"""
    
    def __init__(self, db_path: str = "fitness_users.db", api_key: Optional[str] = None):
        """
        Initialize user matcher
        
        Args:
            db_path: Path to user database
            api_key: DeepInfra API key for embedding generation
        """
        self.database = UserDatabase(db_path)
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
                "goals": profile.goals,
                "fitness_level": profile.fitness_level,
                "preferred_activities": profile.preferred_activities,
                "location": profile.location,
                "age": profile.age,
                "schedule": profile.schedule
            }
            match_details["matches"].append(match_info)
        
        return match_details

def main():
    """Usage example and testing"""
    try:
        # Initialize matcher
        matcher = FitnessUserMatcher()
        
        # Create sample users for database (simulate existing users)
        sample_users = [
            UserProfile(
                user_id="user_001",
                goals="Weight loss and cardio improvement",
                weight=75.0,
                height=170.0,
                age=25,
                fitness_level="Beginner",
                preferred_activities=["Running", "Cycling"],
                schedule="Weekday evenings",
                location="New York",
                additional_info="Looking for motivation partner"
            ),
            UserProfile(
                user_id="user_002",
                goals="Muscle building and strength training",
                weight=80.0,
                height=180.0,
                age=30,
                fitness_level="Intermediate",
                preferred_activities=["Weight lifting", "CrossFit"],
                schedule="Morning workouts",
                location="Los Angeles",
                additional_info="Experienced lifter"
            ),
            UserProfile(
                user_id="user_003",
                goals="Weight loss and muscle toning",
                weight=65.0,
                height=165.0,
                age=28,
                fitness_level="Beginner",
                preferred_activities=["Yoga", "Running"],
                schedule="Weekend workouts",
                location="Chicago",
                additional_info="New to fitness"
            )
        ]
        
        # Add sample users to database (only if database is empty)
        if matcher.database.get_user_count() == 0:
            logger.info("Adding sample users to database...")
            for user in sample_users:
                matcher.add_user_to_database(user)
        
        # Create new user for matching
        new_user = UserProfile(
            user_id="new_user_001",
            goals="Fat loss and improve cardiovascular health",
            weight=72.0,
            height=168.0,
            age=26,
            fitness_level="Beginner",
            preferred_activities=["Running", "Swimming"],
            schedule="Evening workouts",
            location="New York",
            additional_info="Want to find workout buddy"
        )
        
        # Find best matches
        logger.info("Finding best matches for new user...")
        matches = matcher.find_best_matches(new_user, top_k=3)
        
        # Display results
        match_details = matcher.get_match_details(matches)
        
        print("\n" + "="*60)
        print("FITNESS MATCHING RESULTS")
        print("="*60)
        print(f"New User: {new_user.user_id}")
        print(f"Goals: {new_user.goals}")
        print(f"Total matches found: {match_details['total_matches']}")
        print("\nTop Matches:")
        print("-"*60)
        
        for match in match_details['matches']:
            print(f"Rank #{match['rank']}")
            print(f"User ID: {match['user_id']}")
            print(f"Similarity Score: {match['similarity_score']:.4f}")
            print(f"Goals: {match['goals']}")
            print(f"Fitness Level: {match['fitness_level']}")
            print(f"Location: {match['location']}")
            print(f"Preferred Activities: {', '.join(match['preferred_activities'])}")
            print("-"*60)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()