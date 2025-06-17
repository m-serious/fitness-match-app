# Fitness Match App

A comprehensive fitness partner matching system that uses AI-powered embeddings to connect users with compatible workout partners and generates personalized group fitness plans.

## Features

- **AI-Powered User Matching**: Uses advanced embedding technology to match users based on fitness goals, experience, location, and preferences
- **Personalized Fitness Plans**: Generates structured workout plans using GPT-4o-mini with JSON output format
- **Group Management**: Creates and manages fitness groups with detailed workout schedules
- **PostgreSQL Database**: Robust data storage for user profiles, embeddings, and group information
- **Comprehensive Integration**: Full workflow from user registration to group plan generation

## Architecture

The application consists of several key components:

1. **Embedding System** (`embedding.py`): Generates vector embeddings for user profiles
2. **Matching Engine** (`match.py`): Finds compatible fitness partners using cosine similarity
3. **Plan Generator** (`plan_generation.py`): Creates personalized fitness plans in JSON format
4. **Group Manager** (`group_manager.py`): Manages fitness groups and database operations
5. **Integration Script** (`run.sh`): Complete workflow automation

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- DeepInfra API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/m-serious/fitness-match-app.git
cd fitness-match-app
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and database URL
```

3. Run the complete workflow:
```bash
chmod +x run.sh
./run.sh
```

### Environment Variables

Create a `.env` file with the following variables:

```env
POSTGRES_URL="your_postgresql_connection_string"
DEEPINFRA_TOKEN="your_deepinfra_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

## Usage

### Command Line Options

```bash
./run.sh [option]

Options:
  full        Run complete workflow (default)
  deps        Install dependencies only
  test-db     Test database connection only
  embedding   Run embedding test only
  matching    Run matching test only
  integration Run complete integration test
  help        Show help message
```

### Basic Workflow

1. **User Profile Creation**: Define user fitness profiles with comprehensive data
2. **Embedding Generation**: Convert profiles to vector embeddings using AI
3. **Partner Matching**: Find compatible fitness partners using similarity matching
4. **Group Plan Generation**: Create structured fitness plans with GPT-4o-mini
5. **Database Storage**: Store all data in PostgreSQL for persistence

### Example Usage

```python
from embedding import UserProfile
from match import FitnessUserMatcher

# Create user profile
user = UserProfile(
    username="john_doe",
    age=28,
    gender="Male",
    location="San Francisco, CA",
    height=180.0,
    weight=75.0,
    experience=2,
    fitness_goals=["Weight Loss", "Muscle Gain"],
    # ... other fields
)

# Initialize matcher and create fitness group
matcher = FitnessUserMatcher()
result = matcher.create_fitness_group(user)

print(f"Group created: {result['group_name']}")
```

## Database Schema

### Users Table (`fitness_users`)
- `username` (VARCHAR, PRIMARY KEY)
- `age`, `gender`, `location` (Required fields)
- `height`, `weight`, `experience` (Physical attributes)
- `fitness_goals` (JSONB array)
- `embedding` (JSONB vector)
- Diet and lifestyle preferences
- Creation timestamp

### Groups Table (`fitness_groups`)
- `group_id` (VARCHAR, PRIMARY KEY)
- `group_name`, `description`, `goal`
- Workout plans for odd/even days
- Member information
- Creation timestamp

## API Structure

### UserProfile Data Structure

```python
@dataclass
class UserProfile:
    username: str
    age: int
    gender: str
    location: str
    height: float
    weight: float
    experience: int
    eat_out_freq: str
    cook_freq: str
    daily_snacks: str
    snack_type: str
    fruit_veg_servings: str
    beverage_choice: str
    diet_preference: str
    fitness_goals: List[str]
    body_fat: Optional[float] = None
    frequency: Optional[int] = None
    struggling_with: str = ""
```

### Generated Group Plan Format

```json
{
  "groupId": "group_001",
  "groupName": "Morning Runners SF",
  "description": "Early morning running group for cardio fitness",
  "goal": "Cardio Fitness",
  "weeklyPlan": {
    "howManyWeeks": "6",
    "oddDayWorkoutPlan": {
      "title": "Trail Running Session",
      "duration": "45 minutes",
      "exercises": ["warm-up", "main workout", "cool-down"],
      "diet": "Pre/post workout nutrition advice"
    },
    "evenDayWorkoutPlan": {
      "title": "Strength Training",
      "duration": "60 minutes",
      "exercises": ["strength exercises"],
      "diet": "Strength training nutrition"
    }
  },
  "memberFullNames": ["User1", "User2"],
  "memberUsernames": ["user1", "user2"]
}
```

## File Structure

```
fitness-match-app/
├── embedding.py           # User profile embedding generation
├── match.py              # User matching and core logic
├── plan_generation.py    # AI-powered fitness plan generation
├── group_manager.py      # Group database management
├── view_groups.py        # Group visualization utility
├── run.sh               # Main execution script
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Dependencies

- `openai>=1.0.0` - GPT API integration
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Machine learning utilities
- `python-dotenv>=0.19.0` - Environment variable management
- `psycopg2-binary>=2.9.0` - PostgreSQL database adapter

## Testing

Run the complete integration test:

```bash
./run.sh integration
```

This will:
1. Initialize the database
2. Generate sample users
3. Create a test user
4. Find matches
5. Generate a fitness group plan
6. Store everything in the database

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check connection string format
   - Ensure database exists

2. **API Key Issues**
   - Verify OpenAI API key is valid
   - Check DeepInfra token permissions
   - Ensure environment variables are loaded

3. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify all files are present

### Debug Mode

Enable detailed logging by setting the environment variable:
```bash
export PYTHONPATH=.
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## Security

- Never commit `.env` files or API keys
- Use environment variables for sensitive data
- Regularly rotate API keys
- Validate all user inputs

## Performance

- Database queries are optimized with proper indexing
- Group plans cached in database for quick retrieval
- Cosine similarity matching is efficient for typical user bases

## Citations
```bibtex
@misc{fitness_match_app_2025,
  title = {Fitness Match App: AI-Powered Fitness Partner Matching System},
  author = {Zijia Liu and Ali Gedawi and Arihant Jain},
  year = {2025},
  url = {https://github.com/m-serious/fitness-match-app},
  note = {GitHub repository}
}
