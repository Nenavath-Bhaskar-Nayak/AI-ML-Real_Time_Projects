"""
Student Performance ML Project - Production Inference Engine
Provides high-level inference capabilities for single student profiles and batch datasets.
"""
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "student_pass_model.pkl"
)

class StudentPredictor:
    """Production predictor for student pass/fail status and risk tiering."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Please run the model training pipeline first."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.expected_features = [
            "Age", "Gender", "Department", "StudyHours", "Attendance",
            "AssignmentScore", "PreviousExamScore", "PracticeTestScore"
        ]

    def _generate_recommendations(self, data: dict, pass_prob: float) -> list:
        """Generate actionable personalized academic advice based on student metrics."""
        recommendations = []
        study_hours = data.get("StudyHours", 0)
        attendance = data.get("Attendance", 0)
        assignment_score = data.get("AssignmentScore", 0)
        previous_exam = data.get("PreviousExamScore", 0)
        practice_test = data.get("PracticeTestScore", 0)

        if study_hours is not None and study_hours < 4.0:
            recommendations.append(f"Increase weekly study hours from {study_hours}h to at least 5.0h/day.")
        if attendance is not None and attendance < 75.0:
            recommendations.append(f"Attendance is at {attendance}%; target minimum 80% to avoid academic penalties.")
        if assignment_score is not None and assignment_score < 60:
            recommendations.append(f"Assignment score ({assignment_score}) is below standard; seek tutoring or TA assistance.")
        if previous_exam is not None and previous_exam < 50:
            recommendations.append(f"Previous exam score ({previous_exam}) indicates prerequisite knowledge gaps.")
        if practice_test is not None and practice_test < 55:
            recommendations.append("Complete additional practice exams before the final assessment.")

        if not recommendations and pass_prob >= 0.8:
            recommendations.append("Excellent academic standing! Maintain consistent study habits.")
        elif not recommendations:
            recommendations.append("Maintain steady study pace and review key concepts regularly.")

        return recommendations

    def _assess_risk_level(self, pass_prob: float) -> str:
        """Assign risk category based on failure probability."""
        if pass_prob >= 0.75:
            return "Low Risk"
        elif pass_prob >= 0.50:
            return "Moderate Risk"
        else:
            return "High Risk"

    def predict_single(self, student_data: dict) -> dict:
        """
        Predict pass/fail outcome for a single student.
        
        Parameters:
            student_data (dict): Dictionary with student feature values.
            
        Returns:
            dict: Structured prediction results including probabilities and recommendations.
        """
        input_df = pd.DataFrame([student_data])
        for col in self.expected_features:
            if col not in input_df.columns:
                input_df[col] = np.nan

        pred_class = int(self.pipeline.predict(input_df)[0])
        probabilities = self.pipeline.predict_proba(input_df)[0]
        pass_prob = float(probabilities[1]) if len(probabilities) > 1 else float(pred_class)
        fail_prob = 1.0 - pass_prob

        risk_level = self._assess_risk_level(pass_prob)
        recommendations = self._generate_recommendations(student_data, pass_prob)

        return {
            "prediction": "Pass" if pred_class == 1 else "Fail",
            "predicted_label": pred_class,
            "pass_probability": round(pass_prob * 100, 2),
            "fail_probability": round(fail_prob * 100, 2),
            "risk_level": risk_level,
            "recommendations": recommendations
        }

    def predict_batch(self, input_data) -> pd.DataFrame:
        """
        Predict pass/fail outcomes for a batch dataset.
        
        Parameters:
            input_data (pd.DataFrame or str): DataFrame or path to CSV file.
            
        Returns:
            pd.DataFrame: Enriched DataFrame with predictions, probabilities, risk tiers, and advice.
        """
        if isinstance(input_data, str):
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame or path string to a CSV.")

        # Ensure all expected columns exist
        eval_df = df.copy()
        for col in self.expected_features:
            if col not in eval_df.columns:
                eval_df[col] = np.nan

        predictions = self.pipeline.predict(eval_df[self.expected_features])
        probabilities = self.pipeline.predict_proba(eval_df[self.expected_features])

        pass_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else predictions.astype(float)
        
        result_df = df.copy()
        result_df["Predicted_Pass"] = predictions
        result_df["Predicted_Status"] = np.where(predictions == 1, "Pass", "Fail")
        result_df["Pass_Probability_%"] = np.round(pass_probs * 100, 2)
        result_df["Risk_Level"] = [self._assess_risk_level(p) for p in pass_probs]

        return result_df


def run_sample_demo(predictor: StudentPredictor):
    """Run a built-in verification showcase with varied test student profiles."""
    sample_students = [
        {
            "StudentID": "DEMO-001",
            "Age": 21,
            "Gender": "Female",
            "Department": "Computer Science",
            "StudyHours": 8.5,
            "Attendance": 94.0,
            "AssignmentScore": 92,
            "PreviousExamScore": 88,
            "PracticeTestScore": 90,
        },
        {
            "StudentID": "DEMO-002",
            "Age": 22,
            "Gender": "Male",
            "Department": "Commerce",
            "StudyHours": 4.5,
            "Attendance": 72.0,
            "AssignmentScore": 64,
            "PreviousExamScore": 58,
            "PracticeTestScore": 60,
        },
        {
            "StudentID": "DEMO-003",
            "Age": 20,
            "Gender": "Male",
            "Department": "Physics",
            "StudyHours": 1.5,
            "Attendance": 52.0,
            "AssignmentScore": 38,
            "PreviousExamScore": 32,
            "PracticeTestScore": 30,
        }
    ]

    print("\n" + "=" * 70)
    print(" STUDENT PERFORMANCE ML PREDICTION ENGINE - DEMONSTRATION")
    print("=" * 70)

    for student in sample_students:
        res = predictor.predict_single(student)
        print(f"\n[Student ID: {student['StudentID']}] - {student['Department']} ({student['Gender']}, {student['Age']}y)")
        print(f"  Study: {student['StudyHours']}h/d | Attendance: {student['Attendance']}% | Avg Score: {np.mean([student['AssignmentScore'], student['PreviousExamScore'], student['PracticeTestScore']]):.1f}")
        print(f"  --> Prediction:       {res['prediction'].upper()}")
        print(f"  --> Pass Likelihood:  {res['pass_probability']}%")
        print(f"  --> Risk Category:    {res['risk_level']}")
        print(f"  --> Recommendations:")
        for rec in res["recommendations"]:
            print(f"      - {rec}")


    print("\n" + "=" * 70)
def main():
    parser = argparse.ArgumentParser(description="Student Pass/Fail Prediction CLI")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model .pkl")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input CSV path for batch inference")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path for batch results")
    parser.add_argument("--sample", action="store_true", help="Run showcase demo with sample students")

    args = parser.parse_args()

    predictor = StudentPredictor(model_path=args.model)

    if args.input:
        print(f"Processing batch predictions for: {args.input}")
        result_df = predictor.predict_batch(args.input)
        
        output_path = args.output or "batch_predictions_result.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Batch inference complete! Results saved to: {output_path}")
        print("\nSummary of Predictions:")
        print(result_df[["StudentID", "Predicted_Status", "Pass_Probability_%", "Risk_Level"]].head(10))
    else:
        # Default or --sample runs demo
        run_sample_demo(predictor)


if __name__ == "__main__":
    main()
