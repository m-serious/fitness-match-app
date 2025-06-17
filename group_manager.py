import psycopg2
import json
import logging
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FitnessGroup:
    """Fitness group data structure"""
    group_id: str
    group_name: str
    description: str
    goal: str
    how_many_weeks: str
    odd_day_title: str
    odd_day_duration: str
    odd_day_exercises: List[str]
    odd_day_diet: str
    even_day_title: str
    even_day_duration: str
    even_day_exercises: List[str]
    even_day_diet: str
    member_full_names: List[str]
    member_usernames: List[str]
    created_at: Optional[str] = None

class FitnessGroupDatabase:
    """Fitness group database manager using PostgreSQL"""
    
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
        """Initialize fitness groups table"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fitness_groups (
                group_id VARCHAR(255) PRIMARY KEY,
                group_name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                goal VARCHAR(100) NOT NULL,
                how_many_weeks VARCHAR(50) NOT NULL,
                odd_day_title VARCHAR(255) NOT NULL,
                odd_day_duration VARCHAR(100) NOT NULL,
                odd_day_exercises JSONB NOT NULL,
                odd_day_diet TEXT NOT NULL,
                even_day_title VARCHAR(255) NOT NULL,
                even_day_duration VARCHAR(100) NOT NULL,
                even_day_exercises JSONB NOT NULL,
                even_day_diet TEXT NOT NULL,
                member_full_names JSONB NOT NULL,
                member_usernames JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Fitness groups database initialized successfully")
    
    def add_group(self, group_plan: Dict) -> str:
        """
        Add new fitness group to database
        
        Args:
            group_plan: Dictionary containing fitness group plan
            
        Returns:
            Group ID of the added group
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Extract data from group plan
            weekly_plan = group_plan['weeklyPlan']
            odd_day = weekly_plan['oddDayWorkoutPlan']
            even_day = weekly_plan['evenDayWorkoutPlan']
            
            # Convert lists to JSON for storage
            odd_exercises_json = json.dumps(odd_day['exercises'])
            even_exercises_json = json.dumps(even_day['exercises'])
            member_names_json = json.dumps(group_plan['memberFullNames'])
            member_usernames_json = json.dumps(group_plan['memberUsernames'])
            
            cursor.execute('''
                INSERT INTO fitness_groups 
                (group_id, group_name, description, goal, how_many_weeks,
                 odd_day_title, odd_day_duration, odd_day_exercises, odd_day_diet,
                 even_day_title, even_day_duration, even_day_exercises, even_day_diet,
                 member_full_names, member_usernames)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (group_id) DO UPDATE SET
                    group_name = EXCLUDED.group_name,
                    description = EXCLUDED.description,
                    goal = EXCLUDED.goal,
                    how_many_weeks = EXCLUDED.how_many_weeks,
                    odd_day_title = EXCLUDED.odd_day_title,
                    odd_day_duration = EXCLUDED.odd_day_duration,
                    odd_day_exercises = EXCLUDED.odd_day_exercises,
                    odd_day_diet = EXCLUDED.odd_day_diet,
                    even_day_title = EXCLUDED.even_day_title,
                    even_day_duration = EXCLUDED.even_day_duration,
                    even_day_exercises = EXCLUDED.even_day_exercises,
                    even_day_diet = EXCLUDED.even_day_diet,
                    member_full_names = EXCLUDED.member_full_names,
                    member_usernames = EXCLUDED.member_usernames
            ''', (
                group_plan['groupId'],
                group_plan['groupName'],
                group_plan['description'],
                group_plan['goal'],
                weekly_plan['howManyWeeks'],
                odd_day['title'],
                odd_day['duration'],
                odd_exercises_json,
                odd_day['diet'],
                even_day['title'],
                even_day['duration'],
                even_exercises_json,
                even_day['diet'],
                member_names_json,
                member_usernames_json
            ))
            
            conn.commit()
            logger.info(f"Fitness group {group_plan['groupId']} added to database")
            return group_plan['groupId']
            
        except Exception as e:
            logger.error(f"Failed to add fitness group: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_all_groups(self) -> List[FitnessGroup]:
        """
        Get all fitness groups from database
        
        Returns:
            List of FitnessGroup objects
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM fitness_groups ORDER BY created_at DESC')
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        groups = []
        for row in rows:
            try:
                # Parse JSON fields
                odd_exercises = json.loads(row[7]) if isinstance(row[7], str) else row[7]
                even_exercises = json.loads(row[11]) if isinstance(row[11], str) else row[11]
                member_names = json.loads(row[13]) if isinstance(row[13], str) else row[13]
                member_usernames = json.loads(row[14]) if isinstance(row[14], str) else row[14]
                
                group = FitnessGroup(
                    group_id=row[0],
                    group_name=row[1],
                    description=row[2],
                    goal=row[3],
                    how_many_weeks=row[4],
                    odd_day_title=row[5],
                    odd_day_duration=row[6],
                    odd_day_exercises=odd_exercises,
                    odd_day_diet=row[8],
                    even_day_title=row[9],
                    even_day_duration=row[10],
                    even_day_exercises=even_exercises,
                    even_day_diet=row[12],
                    member_full_names=member_names,
                    member_usernames=member_usernames,
                    created_at=row[15].isoformat() if row[15] else None
                )
                groups.append(group)
                
            except Exception as e:
                logger.error(f"Error processing group {row[0]}: {e}")
                continue
        
        logger.info(f"Retrieved {len(groups)} fitness groups from database")
        return groups
    
    def get_groups_by_user(self, username: str) -> List[FitnessGroup]:
        """
        Get all fitness groups that include a specific user
        
        Args:
            username: Username to search for
            
        Returns:
            List of FitnessGroup objects containing the user
        """
        all_groups = self.get_all_groups()
        user_groups = []
        
        for group in all_groups:
            if username in group.member_usernames:
                user_groups.append(group)
        
        logger.info(f"Found {len(user_groups)} groups for user {username}")
        return user_groups
    
    def get_group_count(self) -> int:
        """Get total number of groups in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM fitness_groups')
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def display_groups_table(self):
        """Display all groups in a formatted table"""
        groups = self.get_all_groups()
        
        if not groups:
            print("No fitness groups found in database.")
            return
        
        print("\n" + "="*120)
        print("FITNESS GROUPS DATABASE")
        print("="*120)
        print(f"{'Group ID':<15} {'Group Name':<25} {'Goal':<15} {'Weeks':<6} {'Members':<30} {'Created':<20}")
        print("-"*120)
        
        for group in groups:
            members_str = ", ".join(group.member_full_names)
            if len(members_str) > 28:
                members_str = members_str[:25] + "..."
            
            created_str = group.created_at[:19] if group.created_at else "Unknown"
            
            print(f"{group.group_id:<15} {group.group_name[:24]:<25} {group.goal:<15} {group.how_many_weeks:<6} {members_str:<30} {created_str:<20}")
        
        print("-"*120)
        print(f"Total groups: {len(groups)}")

def main():
    """Usage example and testing"""
    try:
        # Initialize group database
        group_db = FitnessGroupDatabase()
        
        # Display current groups
        group_db.display_groups_table()
        
        # Example group plan (this would normally come from FitnessPlanGenerator)
        example_group_plan = {
            "groupId": "group_test_001",
            "groupName": "SF Morning Runners",
            "description": "Early morning running group for weight loss and cardio fitness in San Francisco",
            "goal": "Cardio Fitness",
            "weeklyPlan": {
                "howManyWeeks": "6",
                "oddDayWorkoutPlan": {
                    "title": "Trail Running Session",
                    "duration": "45 minutes",
                    "exercises": [
                        "5-minute warm-up walk",
                        "30-minute trail run at moderate pace",
                        "5-minute cool-down walk",
                        "5-minute stretching routine"
                    ],
                    "diet": "Pre-workout: banana and water. Post-workout: protein shake with berries"
                },
                "evenDayWorkoutPlan": {
                    "title": "Strength & Cross Training",
                    "duration": "60 minutes",
                    "exercises": [
                        "10-minute dynamic warm-up",
                        "20-minute bodyweight strength circuit",
                        "20-minute cross-training activities",
                        "10-minute cool-down and flexibility"
                    ],
                    "diet": "Pre-workout: light snack 30min before. Post-workout: balanced meal with protein and carbs"
                }
            },
            "memberFullNames": ["Alice Johnson", "Bob Smith"],
            "memberUsernames": ["alice_runner", "bob_lifter"]
        }
        
        # Add example group
        group_id = group_db.add_group(example_group_plan)
        print(f"\nAdded example group: {group_id}")
        
        # Display updated table
        group_db.display_groups_table()
        
        # Test user-specific search
        user_groups = group_db.get_groups_by_user("alice_runner")
        print(f"\nGroups for alice_runner: {len(user_groups)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()