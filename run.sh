#!/bin/bash
# Fitness Match App - Complete Workflow Script
# This script runs the entire fitness matching pipeline:
# 1. Generate user embeddings
# 2. Populate database with sample users
# 3. Find matches for a new user
# 4. Generate personalized fitness plan

echo "=================================="
echo "FITNESS MATCH APP - FULL WORKFLOW"
echo "=================================="

# Set environment variables with corrected database URL
export POSTGRES_URL=""
export DEEPINFRA_TOKEN=""
export OPENAI_API_KEY=""

# Alternative database URLs (try these if the main one fails)
export POSTGRES_URL_BACKUP1=""
export POSTGRES_URL_BACKUP2=""

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Function to install dependencies if needed
install_dependencies() {
    echo ""
    echo "📦 Checking and installing dependencies..."
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo "Creating requirements.txt..."
        cat > requirements.txt << EOF
openai>=1.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
python-dotenv>=0.19.0
psycopg2-binary>=2.9.0
EOF
    fi
    
    # Install dependencies
    pip install -r requirements.txt
    check_status "Dependencies installation"
}

# Function to test database connection with multiple URLs
test_database() {
    echo ""
    echo "🔌 Testing database connection..."
    
    # Array of database URLs to try
    declare -a db_urls=(
        "$POSTGRES_URL"
        "$POSTGRES_URL_BACKUP1" 
        "$POSTGRES_URL_BACKUP2"
    )
    
    for i in "${!db_urls[@]}"; do
        echo "Trying connection URL #$((i+1))..."
        
        if python3 -c "
import psycopg2
import os
import sys
try:
    conn = psycopg2.connect('${db_urls[i]}')
    cursor = conn.cursor()
    cursor.execute('SELECT version();')
    version = cursor.fetchone()
    print(f'✅ Database connected successfully: {version[0][:50]}...')
    cursor.close()
    conn.close()
    # Update the main environment variable with working URL
    print('SUCCESS')
    sys.exit(0)
except Exception as e:
    print(f'❌ Connection attempt failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
            # Update the main POSTGRES_URL with the working one
            export POSTGRES_URL="${db_urls[i]}"
            echo "✅ Database connection established with URL #$((i+1))"
            return 0
        fi
    done
    
    echo "❌ All database connection attempts failed"
    echo "Please check your database credentials and network connectivity"
    
    # Show debugging information
    echo ""
    echo "🔍 Debug information:"
    echo "URL 1: ${db_urls[0]}"
    echo "URL 2: ${db_urls[1]}"
    echo "URL 3: ${db_urls[2]}"
    
    exit 1
}

# Function to run embedding generation and test
run_embedding_test() {
    echo ""
    echo "🔗 Step 1: Testing embedding generation..."
    echo "----------------------------------------"
    
    python3 embedding.py
    check_status "Embedding generation test"
}

# Function to populate database and run matching
run_matching() {
    echo ""
    echo "🎯 Step 2: Populating database and running user matching..."
    echo "---------------------------------------------------------"
    
    python3 match.py
    check_status "User matching and database population"
}

# Function to generate fitness plan
run_plan_generation() {
    echo ""
    echo "📋 Step 3: Generating personalized fitness plan with o4-mini..."
    echo "--------------------------------------------------------------"
    
    python3 plan_generatioin.py
    check_status "Fitness plan generation"
}

# Function to run complete integration test
run_integration_test() {
    echo ""
    echo "🔄 Step 4: Running complete integration test with group creation..."
    echo "----------------------------------------------"
    
    python3 -c "
import sys
sys.path.append('.')
from embedding import UserProfile, FitnessEmbeddingGenerator
from match import FitnessUserMatcher
from plan_generation import FitnessPlanGenerator
from group_manager import FitnessGroupDatabase
import logging

logging.basicConfig(level=logging.INFO)

try:
    print('🚀 Starting complete integration test with group creation...')
    
    # Initialize all components
    matcher = FitnessUserMatcher()
    
    # Ensure database has sample data
    matcher.populate_sample_data()
    
    # Create a new user
    new_user = UserProfile(
        username='integration_test_user',
        age=26,
        gender='Female',
        location='Seattle, WA',
        height=175.0,
        weight=68.0,
        experience=2,
        eat_out_freq='2–3',
        cook_freq='4–5',
        daily_snacks='1',
        snack_type='Healthy (fruit/nuts)',
        fruit_veg_servings='4–5',
        beverage_choice='Mostly water',
        diet_preference='Omnivore',
        fitness_goals=['Weight Loss', 'Muscle Gain', 'Strength Training'],
        body_fat=16.5,
        frequency=4,
        struggling_with='Maintaining consistency and balancing work-life schedule'
    )
    
    print(f'👤 Created test user: {new_user.username}')
    
    # Create fitness group (this includes matching, plan generation, and database storage)
    print('🏃‍♀️ Creating fitness group...')
    group_result = matcher.create_fitness_group(new_user)
    
    if not group_result['success']:
        print(f'❌ Group creation failed: {group_result[\"message\"]}')
        sys.exit(1)
    
    print(f'✅ Fitness group created successfully!')
    print(f'🏷️  Group ID: {group_result[\"group_id\"]}')
    print(f'👥 Group Name: {group_result[\"group_name\"]}')
    print(f'🤝 Members: {group_result[\"primary_user\"]} + {group_result[\"matched_user\"]}')
    print(f'🎯 Similarity Score: {group_result[\"similarity_score\"]:.4f}')
    print(f'📄 Plan saved to: {group_result[\"plan_file\"]}')
    
    # Display all groups in database
    print('\\n📊 Current fitness groups in database:')
    matcher.display_all_groups()
    
    # Show user's groups
    user_groups = matcher.get_user_groups(new_user.username)
    print(f'\\n👤 Groups for {new_user.username}: {len(user_groups)}')
    for group in user_groups:
        print(f'  - {group[\"group_name\"]} (Goal: {group[\"goal\"]}, Duration: {group[\"weeks\"]} weeks)')
    
    print(f'\\n🎉 Integration test completed successfully!')
    
except Exception as e:
    print(f'❌ Integration test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    check_status "Complete integration test with group creation"
}

# Function to display summary
show_summary() {
    echo ""
    echo "📊 WORKFLOW SUMMARY"
    echo "==================="
    echo "✅ Database connection established"
    echo "✅ User embeddings generated using Qwen3-Embedding-4B"
    echo "✅ Sample users populated in PostgreSQL database"
    echo "✅ User matching algorithm tested"
    echo "✅ Fitness plan generated using 4o-mini model"
    echo "✅ Complete integration test passed"
    echo ""
    echo "🎉 Fitness Match App is fully operational!"
    echo ""
    echo "Generated files:"
    echo "- integration_test_fitness_plan.txt (Sample fitness plan)"
    if [ -f "fitness_plan_user_001_user_002.txt" ]; then
        echo "- fitness_plan_user_001_user_002.txt (Test plan)"
    fi
    echo ""
    echo "💡 Next steps:"
    echo "1. Integrate with your web frontend"
    echo "2. Add user authentication"
    echo "3. Implement real-time matching"
    echo "4. Add location-based filtering"
    echo ""
    echo "🔗 Database URL used: $POSTGRES_URL"
}

# Function to display usage help
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  full        Run complete workflow (default)"
    echo "  deps        Install dependencies only"
    echo "  test-db     Test database connection only"
    echo "  embedding   Run embedding test only"
    echo "  matching    Run matching test only"
    echo "  plan        Run plan generation only"
    echo "  integration Run integration test only"
    echo "  help        Show this help message"
    echo "  debug       Show debug information"
}

# Function to show debug information
show_debug() {
    echo "🔍 DEBUG INFORMATION"
    echo "==================="
    echo "Current working directory: $(pwd)"
    echo "Python version: $(python3 --version)"
    echo "Pip version: $(pip --version)"
    echo ""
    echo "Environment variables:"
    echo "POSTGRES_URL: $POSTGRES_URL"
    echo "DEEPINFRA_TOKEN: ${DEEPINFRA_TOKEN:0:10}..."
    echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
    echo ""
    echo "Files in current directory:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    echo ""
    echo "Requirements.txt content:"
    cat requirements.txt 2>/dev/null || echo "requirements.txt not found"
}

# Main execution logic
main() {
    case "${1:-full}" in
        "deps")
            install_dependencies
            ;;
        "test-db")
            test_database
            ;;
        "embedding")
            install_dependencies
            test_database
            run_embedding_test
            ;;
        "matching")
            install_dependencies
            test_database
            run_matching
            ;;
        "plan")
            install_dependencies
            run_plan_generation
            ;;
        "integration")
            install_dependencies
            test_database
            run_integration_test
            ;;
        "full")
            echo "🚀 Starting complete fitness match app workflow..."
            install_dependencies
            test_database
            run_embedding_test
            run_matching
            run_plan_generation
            run_integration_test
            show_summary
            ;;
        "debug")
            show_debug
            ;;
        "help")
            show_help
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Set script permissions and run
chmod +x "$0"

# Start execution
echo "Starting at: $(date)"
main "$@"
echo "Completed at: $(date)"