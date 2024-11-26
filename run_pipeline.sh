#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for this run
timestamp=$(date +%Y%m%d_%H%M%S)

# Define log files
classifier_log="logs/classifier_${timestamp}.json"
final_output="logs/final_${timestamp}.json"
timing_log="logs/timing_${timestamp}.txt"
error_log="logs/errors_${timestamp}.log"

echo "Starting pipeline run at $(date)" | tee "$timing_log"

# Run the pipeline with timing
{
    echo "Classification started at $(date)" | tee -a "$timing_log"
    SECONDS=0
    cat examples/long_statements.json | python classifier.py 2> >(tee -a "$error_log" >&2) | tee "$classifier_log" | \
    {
        echo "Classification completed in $SECONDS seconds" | tee -a "$timing_log"
        echo "Statement generation started at $(date)" | tee -a "$timing_log"
        SECONDS=0
        python statement_generator.py 2> >(tee -a "$error_log" >&2) | tee "$final_output"
        echo "Statement generation completed in $SECONDS seconds" | tee -a "$timing_log"
    }
}

# Calculate and log total time
echo "Pipeline completed at $(date)" | tee -a "$timing_log"
echo "----------------------------------------" >> "$timing_log"

# Print summary
echo "Run completed. Check logs:"
echo "- Classifier output: $classifier_log"
echo "- Final output: $final_output"
echo "- Timing information: $timing_log"
echo "- Errors log: $error_log"

# Show the last few lines of the error log to confirm initialization
echo -e "\nInitialization messages:"
tail -n 5 "$error_log" 