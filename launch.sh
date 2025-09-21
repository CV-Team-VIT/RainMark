#!/bin/bash
# Launch script for Rain Streak Detection System

echo "🌧️ Starting Rain Streak Detection System..."
echo "Opening web browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Start Streamlit app
streamlit run app.py --server.port=8501 --server.address=localhost
