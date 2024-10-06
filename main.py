from src.pipelines.data_ingestion import run_data_ingestion
from src.pipelines.data_validation import run_data_validation
from src.pipelines.data_transformation import run_data_transformation
from src.pipelines.model_training import run_model_training
from src.pipelines.model_evaluation import run_model_evaluation

def main():
    # Run Data Ingestion
    run_data_ingestion()
    
    # Run Data Validation
    run_data_validation()
    
    # Run Data Transformation
    run_data_transformation()
    
    # Run Model Training
    run_model_training()
    
    # Run Model Evaluation
    run_model_evaluation()

if __name__ == "__main__":
    main()
